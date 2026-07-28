/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    process::ExitCode,
    thread::available_parallelism,
};

use clap::Parser;
use hex::FromHex;
use lofty::{
    config::WriteOptions,
    file::TaggedFileExt,
    probe::Probe as LoftyProbe,
    tag::{ItemKey, ItemValue, Tag, TagExt, TagItem},
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rgain::{AppError, AppResult, CacheEntry, GlobalCache, PipelineConfig, Signature, TrackResult, run_pipeline};
use serde::{Serializer, ser::SerializeMap};
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

/// 原子化加载缓存：发生任何错误均视为冷启动（返回空表）
fn load_cache() -> GlobalCache {
    let Ok(file) = File::open(".rgain_cache.json") else { return HashMap::new() };

    let reader = BufReader::new(file);
    // 先将 JSON 解析为带 String Key 的临时字典
    let string_map: HashMap<String, CacheEntry> = serde_json::from_reader(reader).unwrap_or_default();

    // 将其无缝映射为高性能的内部类型
    string_map
        .into_par_iter()
        .filter_map(|(k, v)| {
            // 直接由 hex 库在内部解析并作为一个完整的 [u8; 32] 标量返回。
            let arr = <[u8; 32]>::from_hex(&k).ok()?;
            Some((arr, v))
        })
        .collect()
}

/// 为原生内部缓存实现自定义的外观序列化器
///
/// 直接将底层的 `[u8; 32]` 字典转换为 JSON 对象，彻底抹杀中间态 `HashMap<String, &CacheEntry>` 的堆分配雪崩。
struct CacheSerializer<'a>(&'a GlobalCache);

impl serde::Serialize for CacheSerializer<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // 向 Serde 申请创建一个 JSON Map
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in self.0 {
            // 在这一步才实时进行 hex 编码并立刻写入流
            // 产生的短命 String 会在进入下一次循环前被立刻丢弃，L1 Cache 完美复用
            map.serialize_entry(&hex::encode(k), v)?;
        }
        map.end()
    }
}

/// 原子化存储缓存：写入临时文件后执行 OS 级重命名替换
fn save_cache(cache: &GlobalCache) -> AppResult<()> {
    let tmp_path = ".rgain_cache.json.tmp";
    let target_path = ".rgain_cache.json";

    let file = File::create(tmp_path)?;
    // 增加 BufWriter 减少 write 系统调用
    let writer = BufWriter::with_capacity(64 * 1024, file);

    // 直接将原生 cache 包装后扔给 serde
    serde_json::to_writer(writer, &CacheSerializer(cache)).map_err(|e| AppError::Io(std::io::Error::other(e)))?;

    std::fs::rename(tmp_path, target_path)?;
    Ok(())
}

/// 命令行执行的主入口调度器
///
/// 本函数负责串联 DSP 核心流水线、数据后处理（Map-Reduce）与 I/O 落地分发。
///
/// # Errors
/// - 核心流水线处理崩溃时返回 `AppError`
/// - 标签写入或 JSON 报告导出产生 I/O 错误时返回 `AppError`
fn execute_cli(cli: Cli, threads: usize) -> AppResult<()> {
    // 静态初始化全局只读环境
    let probe = symphonia::default::get_probe();
    let codecs = symphonia::default::get_codecs();
    let mut cache = load_cache();

    let config = PipelineConfig { offset: cli.offset, write_mode: cli.write, probe, codecs, cache: &cache };

    // 触发 DSP 核心并发流水线，获取客观真实的物理增益
    let mut results = run_pipeline(cli.input, threads, &config)?;

    if results.is_empty() {
        tracing::warn!("未找到任何受支持的音频文件。");
        return Ok(());
    }

    // 将新计算出的生数据（Raw）并入缓存树
    for track in &results {
        cache.entry(track.signature).or_insert_with(|| CacheEntry { raw_lufs: track.raw_lufs, peak: track.peak, last_applied_offset: None });
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

    // 下发写操作：只有 needs_write == true 的任务才会被发送到底层
    if cli.write {
        let success_signatures = write_tags_concurrently(&results);

        // 仅为成功写入磁盘的记录更新最后使用过的 offset
        for sig in success_signatures {
            if let Some(entry) = cache.get_mut(&sig) {
                entry.last_applied_offset = cli.offset;
            }
        }
    }

    // 缓存脏页刷盘
    save_cache(&cache)?;

    Ok(())
}

/// 并发写入模式：将算出的增益注入到音频文件的元数据 (ID3v2/Vorbis) 中。
fn write_tags_concurrently(results: &[TrackResult]) -> Vec<Signature> {
    tracing::info!("开始并发写入 {} 个文件的元数据...", results.len());

    let (success_signatures, failed_writes): (Vec<Signature>, usize) = results
        .par_iter()
        .fold(
            || (Vec::with_capacity(128), 0_usize),
            |mut acc, track| {
                // 如果缓存判定该文件无需覆写(偏移量未变)，直接将其视为“写入成功”，透传签名以维持缓存生命周期，并提前结束本帧流水线
                if !track.needs_write {
                    acc.0.push(track.signature);
                    return acc;
                }

                // 执行真正的物理脏页刷盘
                match write_single_track_metadata(track) {
                    Ok(()) => {
                        acc.0.push(track.signature);
                    }
                    Err(e) => {
                        // 故障物理隔离：单文件由于 权限/只读属性 报错，不波及全局线程池调度
                        tracing::error!("写入失败 [{}]: {e:?}", track.path.display());
                        acc.1 += 1;
                    }
                }

                acc
            },
        )
        .reduce(
            || (Vec::new(), 0_usize),
            |mut a, mut b| {
                a.0.append(&mut b.0);
                a.1 += b.1;
                a
            },
        );

    if failed_writes > 0 {
        tracing::warn!("完成写入，但有 {failed_writes} 个文件由于权限或格式损坏写入失败。");
    } else {
        tracing::info!("所有文件元数据写入成功！");
    }

    success_signatures
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
    let mut tagged_file = LoftyProbe::open(&track.path)
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
