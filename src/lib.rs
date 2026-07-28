/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

mod error;
mod file_ops;
mod pipeline;

pub use error::{AppError, AppResult};
pub use pipeline::{run_pipeline, TrackResult, PipelineConfig};

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 缓存条目定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub raw_lufs: f32,
    pub peak: f32,
    /// 记录最后一次 成功写入 时的偏移量。
    /// 如果用户下次以相同的偏移量执行，直接跳过物理写入。
    pub last_applied_offset: Option<f32>,
}

pub type Signature = [u8; 32];
pub type GlobalCache = HashMap<Signature, CacheEntry>;
