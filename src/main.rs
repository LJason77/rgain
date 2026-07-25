/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    process::ExitCode,
    thread::available_parallelism,
};

use clap::Parser;
use lofty::{
    config::WriteOptions,
    file::TaggedFileExt,
    probe::Probe,
    tag::{ItemKey, ItemValue, Tag, TagExt, TagItem},
};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rgain::{AppError, AppResult, TrackResult, run_pipeline};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Debug, Parser)]
#[command(
    name = "rgain",
    about = "ReplayGain Scanner Pipeline",
    long_about = "一个基于广电级 EBU R128 标准的多线程音频回放增益(ReplayGain)计算与写入工具。"
)]
pub struct Cli {
    /// 目标扫描目录
    #[arg(short, long, default_value = ".")]
    pub input: PathBuf,
    /// 强制覆盖的手动增益偏移量(dB)
    #[arg(short, long)]
    pub offset: Option<f32>,
    /// 是否开启写入模式(默认 false：仅输出分析报告而不修改文件)
    #[arg(short, long, default_value_t = false)]
    pub write: bool,
    /// 线程数(默认 0 表示自动检测 CPU 物理核心)
    #[arg(short, long, default_value_t = 0)]
    pub threads: usize,
}

/// 程序的入口点
fn main() -> ExitCode {
    let cli = Cli::parse();

    #[allow(clippy::expect_used)]
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_timer(fmt::time::OffsetTime::local_rfc_3339().expect("无法获得本地时差！"))
        .init();

    // 提前计算并锁定物理并发度
    let threads = if cli.threads > 0 { cli.threads } else { available_parallelism().map_or(1, std::num::NonZero::get) };
    tracing::info!("已锁定 {threads} 个物理线程。");

    let start = std::time::Instant::now();
    match execute_cli(cli, threads) {
        Ok(()) => {
            tracing::info!("完成扫描，耗时 {:.2} 秒。", start.elapsed().as_secs_f32());
            ExitCode::SUCCESS
        }
        Err(e) => {
            // 在产生致命错误时，仅向 stderr 标准错误流输出，保证 stdout 的纯净
            tracing::error!("致命错误: {e:?}");
            ExitCode::FAILURE
        }
    }
}

/// 命令行执行的主入口调度器
///
/// 本函数负责串联 DSP 核心流水线、数据后处理（Map-Reduce）与 I/O 落地分发。
///
/// # Errors
/// - 核心流水线处理崩溃时返回 `AppError`
/// - 标签写入或 JSON 报告导出产生 I/O 错误时返回 `AppError`
fn execute_cli(cli: Cli, threads: usize) -> AppResult<()> {
    // 触发 DSP 核心并发流水线，获取客观真实的物理增益
    let mut results = run_pipeline(cli.input, threads)?;

    if results.is_empty() {
        tracing::warn!("未找到任何受支持的音频文件。");
        return Ok(());
    }

    let offset = cli.offset.unwrap_or(0.0_f32);
    // 音频曲库数量远低于 u32::MAX，先转 u32 再无损转 f64
    let len = f64::from(u32::try_from(results.len()).unwrap_or(u32::MAX));

    // 进行原地并发修改与求和
    let sum_linear_gain: f64 = results
        .par_iter_mut()
        .map(|track| {
            track.track_gain += offset;
            // 将 dB 转为线性振幅乘数 (Linear Amplitude Multiplier)
            let gain_f64 = f64::from(track.track_gain);
            10.0_f64.powf(gain_f64 / 20.0)
        })
        .sum();

    // 计算平均增益
    {
        // 计算线性平均值，再转回对数标尺
        let avg_linear_gain = sum_linear_gain / len;
        let true_avg_gain_db = 20.0 * avg_linear_gain.log10();
        tracing::info!("共处理 {len} 首歌曲, 平均增益: {:.2} dB", true_avg_gain_db);
    }

    export_json_report(&results)?;

    // 根据用户参数进入 I/O 分支
    if cli.write {
        write_tags_concurrently(&results);
    }

    Ok(())
}

/// 并发写入模式：将算出的增益注入到音频文件的元数据 (ID3v2/Vorbis) 中。
fn write_tags_concurrently(results: &[TrackResult]) {
    tracing::info!("开始并发写入 {} 个文件的元数据...", results.len());

    let failed_writes: usize = results
        .par_iter()
        .map(|track| {
            if let Err(e) = write_single_track_metadata(track) {
                tracing::error!("写入失败 [{}]: {e:?}", track.path.display());
                // 计入失败总数
                1_usize
            } else {
                // 成功则为 0
                0_usize
            }
        })
        .sum();

    if failed_writes > 0 {
        tracing::warn!("完成写入，但有 {failed_writes} 个文件由于权限或格式损坏写入失败。");
    } else {
        tracing::info!("所有文件元数据写入成功！");
    }
}

/// 执行单文件的物理元数据写入 (隔离的底层 I/O 操作)
///
/// # Errors
/// 发生以下情况时返回 `AppError`：
/// - 文件被其他进程独占锁定 (`AppError::Io`)。
/// - 底层容器结构破损导致 `lofty` 无法解析文件流 (`AppError::Decode`)。
/// - OS 拒绝写入或磁盘空间不足 (`AppError::Io`)。
#[inline]
fn write_single_track_metadata(track: &TrackResult) -> AppResult<()> {
    // 文件嗅探与流加载：从 VFS 获取文件句柄并交由 Lofty 解析
    let mut tagged_file = Probe::open(&track.path)
        .map_err(|e| AppError::Io(std::io::Error::other(e)))?
        .read()
        .map_err(|e| AppError::Decode(format!("元数据解析失败: {e}")))?;

    // 标签提取策略: 获取该音频格式的推荐主标签类型 (如 MP3->ID3v2, FLAC->Vorbis)
    let tag_type = tagged_file.primary_tag_type();

    let mut tag = tagged_file.remove(tag_type).unwrap_or_else(|| Tag::new(tag_type));

    // 内存覆盖写
    // Gain 协议规范: 带正负号的浮点数，保留两位小数，后跟 " dB"
    let gain_str = format!("{:+.2} dB", track.track_gain);
    tag.insert(TagItem::new(ItemKey::ReplayGainTrackGain, ItemValue::Text(gain_str)));

    // Peak 协议规范: 绝对浮点数，保留六位小数
    let peak_str = format!("{:.6}", track.peak);
    tag.insert(TagItem::new(ItemKey::ReplayGainTrackPeak, ItemValue::Text(peak_str)));

    // OS 物理刷盘(fsync/pwrite)
    // 注意：如果新增的文本溢出了原有的 Padding 预留空间，这里会触发操作系统的全文件重写。
    tag.save_to_path(&track.path, WriteOptions::default()).map_err(|e| AppError::Io(std::io::Error::other(e)))?;

    Ok(())
}

/// 只读报告模式：将扫描结果无损序列化为 JSON
///
/// # 内存与系统调用优化
/// 不使用 `serde_json::to_string(&results)` 这种将所有结果一次性怼进内存 `String` 的操作！
/// 而是利用 `BufWriter` 直接对接 `File` 描述符，实现**流式序列化(Streaming Serialization)**。
fn export_json_report(results: &[TrackResult]) -> AppResult<()> {
    let report_path = "rgain_report.json";
    let file = File::create(report_path)?;

    // 分配一块 64KB 的连续内存充当缓冲区。
    // 这将把成千上万次碎散的 write 系统调用合并成极少数的几次大块内存物理刷新，
    // 彻底消灭因频繁陷入 OS 内核态而引发的吞吐量断崖。
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    serde_json::to_writer(&mut writer, results).map_err(|e| AppError::Io(std::io::Error::other(e)))?;

    // 强制将缓冲区残留数据刷入磁盘，避免掉电丢失
    writer.flush()?;

    tracing::info!("✅ 扫描完成！增益报告已导出至: {report_path}");

    Ok(())
}
