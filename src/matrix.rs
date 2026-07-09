use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const CTX_MARGIN: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Case {
    pub parallel_size: usize,
    pub num_prompts: usize,
    pub input_len: usize,
    pub output_len: usize,
    pub input_bucket: String,
    pub output_bucket: String,
    pub concurrency_bucket: String,
}

impl Case {
    pub fn case_id(&self) -> String {
        format!(
            "p{}-c{}-{}in-{}out-i{}-o{}",
            self.parallel_size,
            self.num_prompts,
            self.input_bucket,
            self.output_bucket,
            self.input_len,
            self.output_len
        )
    }

    pub fn required_context_per_slot(&self) -> usize {
        self.input_len + self.output_len + CTX_MARGIN
    }
}

pub fn parse_range(value: &str, name: &str) -> Result<(usize, usize)> {
    let normalized = value
        .trim()
        .replace('~', "-")
        .replace('，', ",")
        .replace(' ', "");
    let parts: Vec<&str> = if normalized.contains('-') {
        normalized.splitn(2, '-').collect()
    } else if normalized.contains(',') {
        normalized.splitn(2, ',').collect()
    } else {
        vec![&normalized, &normalized]
    };
    let lo = parts[0].parse::<usize>()?;
    let hi = parts[1].parse::<usize>()?;
    if lo == 0 || hi == 0 || lo > hi {
        bail!("{} 范围必须是正整数且左值 <= 右值: {}", name, value);
    }
    Ok((lo, hi))
}

pub fn parse_int_list(value: &str, name: &str) -> Result<Vec<usize>> {
    let normalized = value.trim().replace('，', ",").replace(' ', "");
    if normalized.is_empty() {
        bail!("{} 列表不能为空", name);
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for part in normalized.split(',') {
        let item = part.parse::<usize>()?;
        if item == 0 {
            bail!("{} 必须是正整数: {}", name, value);
        }
        if seen.insert(item) {
            out.push(item);
        }
    }
    Ok(out)
}

pub fn parse_gpu_devices(value: &str) -> Result<(String, Vec<String>)> {
    let normalized = value
        .trim()
        .replace('，', ",")
        .replace(' ', "")
        .to_lowercase();
    if normalized.is_empty() || normalized == "all" {
        return Ok(("all".to_string(), vec![]));
    }
    if normalized == "none" || normalized == "cpu" {
        return Ok(("none".to_string(), vec!["none".to_string()]));
    }
    let mut devices = Vec::new();
    for part in normalized.split(',') {
        if part.is_empty() {
            continue;
        }
        if part.contains('-') && !part.to_uppercase().starts_with("CUDA") {
            let mut hp = part.splitn(2, '-');
            let lo = hp.next().unwrap().parse::<usize>()?;
            let hi = hp.next().unwrap().parse::<usize>()?;
            if lo > hi {
                bail!("GPU 设备范围错误: {}", value);
            }
            for i in lo..=hi {
                devices.push(format!("CUDA{}", i));
            }
        } else if part.chars().all(|c| c.is_ascii_digit()) {
            devices.push(format!("CUDA{}", part));
        } else {
            devices.push(part.to_uppercase());
        }
    }
    if devices.is_empty() {
        bail!("GPU 设备格式错误: {}", value);
    }
    let label = devices.join(",");
    Ok((label, devices))
}

fn powers_of_two_between(lo: usize, hi: usize) -> Vec<usize> {
    let mut vals = Vec::new();
    let mut n = 1usize;
    while n < lo {
        n *= 2;
    }
    while n <= hi {
        vals.push(n);
        n *= 2;
    }
    vals
}

fn sample_evenly(values: &[usize], limit: usize) -> Vec<usize> {
    if values.len() <= limit {
        return values.to_vec();
    }
    if limit <= 1 {
        return vec![values[0]];
    }
    let mut selected = Vec::new();
    for i in 0..limit {
        let idx = (i * (values.len() - 1)) as f64 / (limit - 1) as f64;
        let idx = idx.round() as usize;
        selected.push(values[idx.min(values.len() - 1)]);
    }
    selected.sort_unstable();
    selected.dedup();
    selected
}

pub fn nice_points(lo: usize, hi: usize, max_points: usize) -> Vec<usize> {
    if lo == hi {
        return vec![lo];
    }
    let mut values: Vec<usize> = [lo, hi]
        .into_iter()
        .chain(powers_of_two_between(lo, hi))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    values.sort_unstable();
    values = sample_evenly(&values, max_points);
    let mut set: HashSet<usize> = values.into_iter().collect();
    set.insert(lo);
    set.insert(hi);
    let mut values: Vec<usize> = set.into_iter().collect();
    values.sort_unstable();
    values
}

pub fn parallel_points(lo: usize, hi: usize) -> Vec<usize> {
    let mut values = HashSet::new();
    if lo <= 1 && 1 <= hi {
        values.insert(1);
    }
    if lo <= 2 && 2 <= hi {
        values.insert(2);
    }
    for n in lo.max(4)..=hi {
        if n % 4 == 0 {
            values.insert(n);
        }
    }
    values.insert(lo);
    values.insert(hi);
    let mut values: Vec<usize> = values.into_iter().collect();
    values.sort_unstable();
    values
}

pub fn prompt_points(lo: usize, hi: usize, max_points: usize) -> Vec<usize> {
    let mut points = nice_points(lo, hi, max_points);
    if lo <= 1 && 1 <= hi && !points.contains(&1) {
        points.insert(0, 1);
    }
    points.sort_unstable();
    points.dedup();
    points
}

pub fn bucket(value: usize, points: &[usize]) -> String {
    if value == *points.iter().min().unwrap_or(&value) {
        "short".to_string()
    } else if value == *points.iter().max().unwrap_or(&value) {
        "long".to_string()
    } else {
        "mid".to_string()
    }
}

pub fn auto_batch_size(max_input_len: usize, max_batch_size: usize) -> usize {
    if max_input_len <= 512 {
        return 512.min(max_batch_size);
    }
    if max_input_len <= 1024 {
        return 1024.min(max_batch_size);
    }
    max_batch_size
}

pub fn grouped_by_required_context(cases: &[Case]) -> Vec<(usize, Vec<Case>)> {
    let mut contexts: Vec<usize> = cases
        .iter()
        .map(|c| c.required_context_per_slot())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    contexts.sort_unstable();
    contexts
        .into_iter()
        .map(|ctx| {
            let group: Vec<Case> = cases
                .iter()
                .filter(|c| c.required_context_per_slot() == ctx)
                .cloned()
                .collect();
            (ctx, group)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nice_points() {
        assert_eq!(nice_points(64, 4096, 4), vec![64, 256, 1024, 4096]);
        assert_eq!(nice_points(512, 50000, 4), vec![512, 2048, 16384, 50000]);
    }

    #[test]
    fn test_parallel_points() {
        assert_eq!(parallel_points(1, 8), vec![1, 2, 4, 8]);
        assert_eq!(parallel_points(1, 4), vec![1, 2, 4]);
    }

    #[test]
    fn test_parse_gpu_devices() {
        assert_eq!(
            parse_gpu_devices("all").unwrap(),
            ("all".to_string(), vec![])
        );
        assert_eq!(
            parse_gpu_devices("none").unwrap(),
            ("none".to_string(), vec!["none".to_string()])
        );
        assert_eq!(
            parse_gpu_devices("0-2").unwrap().1,
            vec!["CUDA0", "CUDA1", "CUDA2"]
        );
        assert_eq!(
            parse_gpu_devices("CUDA0,CUDA1").unwrap().1,
            vec!["CUDA0", "CUDA1"]
        );
        assert_eq!(
            parse_gpu_devices("0,2,4").unwrap().1,
            vec!["CUDA0", "CUDA2", "CUDA4"]
        );
        assert!(parse_gpu_devices("2-1").is_err());
    }

    #[test]
    fn test_parse_range() {
        assert_eq!(parse_range("64-4096", "x").unwrap(), (64, 4096));
        assert_eq!(parse_range("512,50000", "x").unwrap(), (512, 50000));
        assert_eq!(parse_range("128", "x").unwrap(), (128, 128));
        assert!(parse_range("0-10", "x").is_err());
        assert!(parse_range("10-5", "x").is_err());
        assert!(parse_range("abc", "x").is_err());
    }

    #[test]
    fn test_parse_int_list() {
        assert_eq!(parse_int_list("1,4,8", "x").unwrap(), vec![1, 4, 8]);
        assert_eq!(parse_int_list("1,1,4", "x").unwrap(), vec![1, 4]);
        assert!(parse_int_list("", "x").is_err());
        assert!(parse_int_list("1,0", "x").is_err());
    }

    #[test]
    fn test_bucket() {
        let points = vec![64, 256, 1024, 4096];
        assert_eq!(bucket(64, &points), "short");
        assert_eq!(bucket(256, &points), "mid");
        assert_eq!(bucket(4096, &points), "long");
    }

    #[test]
    fn test_auto_batch_size() {
        assert_eq!(auto_batch_size(256, 2048), 512);
        assert_eq!(auto_batch_size(512, 2048), 512);
        assert_eq!(auto_batch_size(768, 2048), 1024);
        assert_eq!(auto_batch_size(1024, 2048), 1024);
        assert_eq!(auto_batch_size(2048, 2048), 2048);
        assert_eq!(auto_batch_size(2048, 512), 512);
    }

    #[test]
    fn test_prompt_points() {
        assert_eq!(prompt_points(1, 32, 3), vec![1, 8, 32]);
        assert_eq!(prompt_points(2, 32, 3), vec![2, 8, 32]);
    }

    #[test]
    fn test_grouped_by_required_context() {
        let cases = vec![
            Case {
                parallel_size: 1,
                num_prompts: 1,
                input_len: 64,
                output_len: 64,
                input_bucket: "short".to_string(),
                output_bucket: "short".to_string(),
                concurrency_bucket: "single".to_string(),
            },
            Case {
                parallel_size: 1,
                num_prompts: 2,
                input_len: 64,
                output_len: 64,
                input_bucket: "short".to_string(),
                output_bucket: "short".to_string(),
                concurrency_bucket: "multi".to_string(),
            },
            Case {
                parallel_size: 1,
                num_prompts: 1,
                input_len: 128,
                output_len: 64,
                input_bucket: "mid".to_string(),
                output_bucket: "short".to_string(),
                concurrency_bucket: "single".to_string(),
            },
        ];
        let grouped = grouped_by_required_context(&cases);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].0, 64 + 64 + CTX_MARGIN);
        assert_eq!(grouped[0].1.len(), 2);
        assert_eq!(grouped[1].0, 128 + 64 + CTX_MARGIN);
    }

    #[test]
    fn test_case_id_and_required_context() {
        let case = Case {
            parallel_size: 2,
            num_prompts: 4,
            input_len: 64,
            output_len: 128,
            input_bucket: "short".to_string(),
            output_bucket: "mid".to_string(),
            concurrency_bucket: "multi".to_string(),
        };
        assert_eq!(case.case_id(), "p2-c4-shortin-midout-i64-o128".to_string());
        assert_eq!(case.required_context_per_slot(), 64 + 128 + CTX_MARGIN);
    }
}
