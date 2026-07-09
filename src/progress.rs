use crate::matrix::Case;
use crate::utils::format_seconds;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::time::Instant;

pub struct ProgressTracker {
    total: usize,
    current: usize,
    started_at: Instant,
    status_counts: HashMap<String, usize>,
    mode: String,
    use_tty: bool,
}

impl ProgressTracker {
    pub fn new(total: usize, mode: String) -> Self {
        let use_tty = std::io::stdout().is_terminal();
        Self {
            total,
            current: 0,
            started_at: Instant::now(),
            status_counts: HashMap::new(),
            mode,
            use_tty,
        }
    }

    pub fn update(&mut self, case: &Case, status: &str) {
        if self.total == 0 || self.mode == "none" {
            return;
        }
        self.current += 1;
        *self.status_counts.entry(status.to_string()).or_insert(0) += 1;
        let msg = self.message(case, status);
        let end = if self.use_tty && self.current < self.total {
            "\r"
        } else {
            "\n"
        };
        print!("{}{}", msg, end);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    pub fn write(&mut self, message: String) {
        if self.mode == "none" {
            return;
        }
        if self.use_tty && self.current < self.total {
            println!();
        }
        println!("{}", message);
    }

    pub fn close(&mut self) {
        if self.total == 0 || self.mode == "none" {
            return;
        }
        if self.use_tty {
            println!();
        }
        let summary: Vec<String> = {
            let mut items: Vec<_> = self.status_counts.iter().collect();
            items.sort_by(|a, b| a.0.cmp(b.0));
            items
                .into_iter()
                .map(|(status, count)| format!("{}={}", status, count))
                .collect()
        };
        println!(
            "进度完成: {}/{} | {}",
            self.current,
            self.total,
            summary.join(", ")
        );
    }

    fn message(&self, case: &Case, status: &str) -> String {
        let elapsed = self.started_at.elapsed().as_secs_f64().max(0.001);
        let rate = self.current as f64 / elapsed;
        let remaining = self.total.saturating_sub(self.current) as f64;
        let eta = if rate > 0.0 { remaining / rate } else { 0.0 };
        let percent = self.current as f64 / self.total as f64 * 100.0;
        format!(
            "进度 {}/{} ({:5.1}%) ETA {} | {} | {}",
            self.current,
            self.total,
            percent,
            format_seconds(eta as u64),
            status,
            case.case_id()
        )
    }
}
