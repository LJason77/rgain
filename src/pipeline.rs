/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

use std::{
    fs::File,
    path::PathBuf,
    thread::{self},
};

use crossbeam_channel::bounded;
use ebur128::{EbuR128, Mode};
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::Serialize;
use symphonia::core::{
    codecs::audio::{AudioDecoder, AudioDecoderOptions},
    common::Limit,
    errors::Error,
    formats::{FormatOptions, FormatReader, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

use crate::{AppError, AppResult, file_ops::traverse_dirs};

/// 单轨处理结果
#[derive(Debug, Serialize)]
pub struct TrackResult {
    pub path: PathBuf,
    pub track_gain: f32,
    pub peak: f32,
}

/// 驱动音频增益扫描的核心流水线
///
/// # Errors
/// 当线程池初始化失败或 IO 扫描线程崩溃时返回 `AppError`
pub fn run_pipeline(input: PathBuf, threads: usize) -> AppResult<Vec<TrackResult>> {
    // 创建物理隔离的局部线程池
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| AppError::Decode(format!("线程池初始化失败: {e}")))?;

    // 建立背压通道 (固定容量 1024，完美平衡内存消耗与 I/O 突发)
    let (tx_path, rx_path) = bounded::<PathBuf>(1024);

    // 启动后台 I/O 目录扫描线程
    let scanner_thread = thread::spawn(move || {
        // 显式忽略 Result：因 traverse_dirs 内部已处理跳过不可读分支，
        // 只有当根目录彻底崩溃或消费者断开时才会 Error 退出，这是符合预期的生命周期结束。
        let _ = traverse_dirs(input, usize::MAX, &tx_path);
    });

    // 在局部线程池中驱动 DSP 计算流水线
    let (results, failed_count) = pool.install(|| {
        rx_path
            .into_iter()
            .par_bridge()
            .fold(
                || (Vec::with_capacity(128), 0_usize),
                |mut acc, path| {
                    match process_audio_file(path.clone()) {
                        Ok(res) => acc.0.push(res),
                        Err(e) => {
                            tracing::warn!("[{}] 处理失败: {e:?}", path.display());
                            acc.1 += 1;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || (Vec::new(), 0_usize),
                |mut a, mut b| {
                    // Vec::append 在底层直接映射为 std::ptr::copy_nonoverlapping，
                    // 它将触发 CPU 的宽向量指令(如 AVX/SIMD)，以极速内存块复制（memcpy）的形式合并数据，
                    // 性能远超显式的 for 循环 push。
                    a.0.append(&mut b.0);
                    a.1 += b.1;
                    a
                },
            )
    });

    // 回收 I/O 线程资源，确保内存屏障同步
    scanner_thread.join().map_err(|_| AppError::Decode("Scanner thread 发生系统级 Panic 崩溃".into()))?;

    // 将收集到的结构体序列化
    tracing::info!("成功处理: {} 首, 失败: {failed_count} 首", results.len());

    Ok(results)
}

/// 计算单首音频文件的 `ReplayGain` 2.0 (LUFS) 回放增益与峰值。
///
/// 构建了从磁盘 Page Cache 到 CPU FPU (浮点运算单元) 的低分配解码流水线。
///
/// # Errors
/// 当发生以下情况时返回 `AppError`：
/// - 文件不存在或因 OS 权限拒绝访问 (`AppError::Io`)。
/// - 格式无法被 Symphonia 嗅探或不支持该编码 (`AppError::Decode`)。
/// - 文件内部损坏或未包含任何有效音频轨 (`AppError::Decode`)。
/// - `EbuR128` 状态机初始化失败（如遇到了不支持的极端采样率）。
fn process_audio_file(path: PathBuf) -> AppResult<TrackResult> {
    // 建立 OS 文件描述符，包裹进 Symphonia 的流式读取器
    let file = File::open(&path)?;
    // 覆盖默认的短缓冲，使用 256KB 连续物理内存吸收磁盘的突发(Burst)读取
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions { buffer_len: 256 * 1024 });

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // 暴力斩断视觉数据的堆分配：
    // 很多无损音乐内嵌了 5MB+ 的高清封面。
    // 将其限制为 0，告诉引擎在解析到图片帧时直接移动文件指针(lseek)跳过，彻底消灭此处的 malloc 开销
    let metadata_opts = MetadataOptions::default().limit_visual_bytes(Limit::Maximum(0));

    // 格式嗅探与探测(避免依赖文件后缀名，直接读取 Magic Number)
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), metadata_opts)
        .map_err(|e| AppError::Decode(format!("格式探测失败: {e}")))?;

    // 不信任 default_track()，直接遍历寻找物理存储上的第一个有效音频轨
    let (track_id, audio_codec_params) = format
        .tracks()
        .iter()
        .find_map(|t| t.codec_params.as_ref().and_then(|p| p.audio()).map(|audio_params| (t.id, audio_params)))
        .ok_or_else(|| AppError::Decode("文件未包含有效音频轨或编解码器参数".into()))?;

    // 初始化对应的音频解码器 (FLAC/MP3/WAV 等)
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(audio_codec_params, &AudioDecoderOptions::default()).map_err(|e| {
        tracing::warn!("{audio_codec_params:?} 无法初始化解码器: {e:?}");
        AppError::Decode(format!("解码器初始化失败: {e}"))
    })?;

    // 提取音频元数据：声道数与采样率
    // 音频通道数不可能超过 u32::MAX，故使用强制转型
    #[allow(clippy::cast_possible_truncation)]
    let channels = audio_codec_params.channels.as_ref().map_or(2, |c| c.count() as u32); // 默认降级为双声道
    let sample_rate = audio_codec_params.sample_rate.unwrap_or(48000);

    // 初始化 EBU R128 DSP 状态机
    // - Mode::I 开启 Integrated Loudness (积分响度/全局门限)
    // - Mode::SAMPLE_PEAK 开启采样峰值计算，相比 TRUE_PEAK 能大幅节省 CPU 时钟周期
    let mut ebu =
        EbuR128::new(channels, sample_rate, Mode::I | Mode::SAMPLE_PEAK).map_err(|e| AppError::Decode(format!("EBUR128 初始化失败: {e:?}")))?;

    let (lufs, peak) = run_dsp_loop(&mut *format, &mut *decoder, &mut ebu, channels, track_id)?;

    // ReplayGain 2.0 工业标准计算
    // 标准化参考基准线为 -18.0 LUFS
    let track_gain = -18.0 - lufs;

    Ok(TrackResult { path, track_gain, peak })
}

/// DSP 核心计算流水线
/// 返回元组： (算出的 LUFS 响度, 全局最大峰值)
#[inline]
fn run_dsp_loop(
    format: &mut dyn FormatReader, decoder: &mut dyn AudioDecoder, ebu: &mut EbuR128, channels: u32, target_track_id: u32,
) -> AppResult<(f32, f32)> {
    // 8192 采样点足以无缝吞下绝大多数编码的单帧数据（双声道 FLAC 一般为 4096*2 = 8192）
    let mut sample_buf: Vec<f32> = Vec::with_capacity(8192);

    loop {
        // 解码流水线 Phase 1：从 IO 抽取 Packet 数据帧
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break, // 文件正常读取完毕，脱离了 Error 的控制流
            Err(Error::ResetRequired) => continue,
            Err(Error::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(AppError::Decode(format!("读取包失败: {err}"))),
        };

        // 通道隔离过滤
        if packet.track_id != target_track_id {
            continue;
        }

        // 解码流水线 Phase 2：将压缩包送入对应算法解码出原始 PCM 数据
        let audio_buf = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(Error::DecodeError(_)) => continue, // 忽略损坏的音频帧
            Err(err) => return Err(AppError::Decode(format!("解码错误: {err}"))),
        };

        // 解码流水线 Phase 3：内存规整化 (0.6 原生直接支持)
        // 获取当前帧的交错样本总数 (Frames * Channels)
        let sample_count = audio_buf.samples_interleaved();
        sample_buf.reserve(sample_count);

        // SAFETY:
        // 1. 前置的 `reserve` 确保了底层物理内存的 capacity 必然 >= sample_count。
        // 2. 紧接着的 `copy_to_slice_interleaved` 会立刻进行全量覆盖写入，
        //    保证这块未初始化内存瞬间被合法音频 f32 填满，DSP 绝不会读到脏数据。
        #[allow(clippy::uninit_vec)]
        unsafe {
            sample_buf.set_len(sample_count);
        }

        // 根据 sample_buf 的泛型 (f32)，自动将底层数据转为 f32 并交错排布
        audio_buf.copy_to_slice_interleaved(&mut sample_buf);

        // DSP 流水线 Phase 4：推入 K-Weighting 滤波器与门限计算器
        ebu.add_frames_f32(&sample_buf).map_err(|_| AppError::Decode("EBUR128 滤波器注入失败".into()))?;
    }

    // 计算全局积分响度 (Integrated Loudness)
    let lufs = ebu.loudness_global().map_err(|_| AppError::Decode("全局响度计算失败(可能音频过短)".into()))?;

    // 扫描各声道最大峰值 (用于防止增益后 Clipping 爆音)
    let peak = (0..channels).filter_map(|c| ebu.sample_peak(c).ok()).fold(0.0_f64, f64::max);

    // 仅在生命周期最后一步向下收窄类型，并显式忽略不可避免的截断警告
    #[allow(clippy::cast_possible_truncation)]
    Ok((lufs as f32, peak as f32))
}
