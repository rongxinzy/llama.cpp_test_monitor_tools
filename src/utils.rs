use std::path::Path;

pub fn now_str() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

pub fn run_id() -> String {
    format!(
        "{}-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        std::process::id()
    )
}

pub fn file_ts() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

pub fn hostname() -> String {
    std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "host".to_string())
}

pub fn tail_file(path: &Path, n: usize) -> String {
    if !path.exists() {
        return String::new();
    }
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let lines: Vec<&str> = text.lines().collect();
            lines
                .iter()
                .rev()
                .take(n)
                .rev()
                .copied()
                .collect::<Vec<_>>()
                .join("\n")
        }
        Err(_) => String::new(),
    }
}

pub fn format_seconds(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    if hours > 0 {
        format!("{}h{:02}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m{:02}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

pub fn normalize_error(error: &str) -> String {
    error
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(500)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_tail_file_nonexistent() {
        assert_eq!(tail_file(Path::new("/does/not/exist"), 10), "");
    }

    #[test]
    fn test_tail_file_existing() {
        let tmp = std::env::temp_dir().join("llama-test-matrix-tail-test.txt");
        let mut f = std::fs::File::create(&tmp).unwrap();
        for i in 0..20 {
            writeln!(f, "line {}", i).unwrap();
        }
        let out = tail_file(&tmp, 5);
        assert_eq!(out, "line 15\nline 16\nline 17\nline 18\nline 19");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_format_seconds() {
        assert_eq!(format_seconds(45), "45s");
        assert_eq!(format_seconds(125), "2m05s");
        assert_eq!(format_seconds(3665), "1h01m");
    }

    #[test]
    fn test_normalize_error() {
        assert_eq!(normalize_error("  a\n\tb   c  "), "a b c");
        assert_eq!(normalize_error(""), "");
        let long = "x ".repeat(300);
        assert_eq!(normalize_error(&long).len(), 500);
    }
}
