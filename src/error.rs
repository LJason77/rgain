/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

/// crate 级结果类型
pub type AppResult<T> = Result<T, AppError>;

/// 错误定义
#[derive(Debug)]
pub enum AppError {
    Io(std::io::Error),
    Decode(String),
    // 后续可扩展更多领域错误
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::Io(err)
    }
}
