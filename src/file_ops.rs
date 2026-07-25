/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

use std::path::{Path, PathBuf};

use crossbeam_channel::Sender;

use crate::AppError;

/// 递归遍历目录，利用 Channel 流式投递文件，实现边扫描边计算。
///
/// # 参数
/// - `path`: 起始路径
/// - `max_depth`: 最大遍历深度
/// - `tx`: 带有背压机制的同步发送通道
///
/// # Errors
/// 仅当根目录完全无法访问时，返回 `AppError`
pub fn traverse_dirs<P: AsRef<Path>>(path: P, max_depth: usize, tx: &Sender<PathBuf>) -> Result<(), AppError> {
    let root = path.as_ref();

    // 致命错误提前拦截：如果根节点就不合法，直接拒绝执行
    if std::fs::read_dir(root).is_err() {
        return Err(AppError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "起始目录不存在或无法访问")));
    }

    // 预分配足够的容量，避免遍历过程中的 Vec 扩容拷贝
    let mut current_path = PathBuf::with_capacity(512);
    current_path.push(root);

    traverse_recursive(&mut current_path, tx, max_depth, 0);

    Ok(())
}

/// 内部递归函数：采用状态机模式原地修改 PathBuf，实现 O(1) 内存分配。
/// 返回 `bool` 控制级联中断：true 继续，false 代表通道已断开需立刻终止全盘扫描。
fn traverse_recursive(current_path: &mut PathBuf, tx: &Sender<PathBuf>, max_depth: usize, current_depth: usize) -> bool {
    if current_depth > max_depth {
        return true;
    }

    // 如果某个子目录因权限无法读取，直接跳过
    let Ok(entries) = std::fs::read_dir(&current_path) else {
        return true;
    };

    for entry_result in entries {
        // 利用 let else 抹平嵌套，过滤读取失败的条目
        let Ok(entry) = entry_result else { continue };
        // 直接读取 d_type (在 Linux/macOS 绕过 stat)
        let Ok(file_type) = entry.file_type() else { continue };

        // 复用同一个堆内存，压入当前文件名
        current_path.push(entry.file_name());

        if file_type.is_dir() {
            // 深度优先递归。如果子调用返回 false，立即向上级联退出
            if !traverse_recursive(current_path, tx, max_depth, current_depth + 1) {
                return false;
            }
        } else if file_type.is_file() && is_supported_audio(current_path) {
            // is_file 顺带过滤掉了符号链接(Symlink)、设备文件(Block/Char)等

            // 仅在确认是目标文件时，才执行全生命周期的 clone (不可避免的跨线程拷贝)
            if tx.send(current_path.clone()).is_err() {
                // 接收端 (Rayon 或 主进程) 已被释放，触发级联熔断，立即终止所有递归
                return false;
            }
        }

        // 状态回溯，弹出当前文件名，维持内存连贯性
        current_path.pop();
    }

    true
}

/// 零分配(Zero-allocation) 的扩展名校验
#[allow(clippy::inline_always)]
#[inline(always)]
fn is_supported_audio(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return false;
    };

    // eq_ignore_ascii_case 避免了调用 .to_lowercase() 带来的堆内存分配
    ext.eq_ignore_ascii_case("flac") || ext.eq_ignore_ascii_case("mp3") || ext.eq_ignore_ascii_case("wav")
}
