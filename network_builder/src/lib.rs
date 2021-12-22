use csv::Writer;
use pyo3::prelude::*;
use ndarray::Array2;
use std::{collections::HashMap};
use std::fs;
use itertools::Itertools;
use serde::{Deserialize};
use indicatif::{ProgressIterator, ProgressBar};

const PB_INC: u64 = 5000;

#[derive(Deserialize)]
struct DataRecord {
    #[serde(rename = "f0_")]
    count: i32,
    subreddit: String,
    author: String,
}

fn load_records(data_path: &str) -> Result<Vec<DataRecord>, csv::Error> {

    let mut all_records = Vec::new();

    let paths = fs::read_dir(data_path).unwrap().collect_vec();

    println!("Reading files...");
    for path in paths.into_iter().progress() {

        let path = path.unwrap().path();
        let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_path(path).unwrap();

        for record in rdr.deserialize() {
            all_records.push(record?);
        }
    }

    return Ok(all_records)
}

fn is_valid_record(record: &DataRecord, min_count_bot_exclusion: i32) -> bool {
    record.author != "[deleted]"
    && record.author != "AutoModerator"
    && !(record.count > min_count_bot_exclusion && record.author.to_lowercase().ends_with("bot"))
}

fn filter_records(records: &mut Vec<DataRecord>, min_count_bot_exclusion: i32, subreddit_comment_threshold: i32) -> &mut Vec<DataRecord> {

    // Exclude deleted accounts and bots.
    records.retain(|record| is_valid_record(record, min_count_bot_exclusion));

    // Count how many comments any given subreddit has.

    println!("Counting subreddit comments");
    let pb = ProgressBar::new(records.len() as u64);
    let mut counts = HashMap::<String, i32>::new();
    let mut progress_count: u64 = 0;
    for r in records.iter() {
        *counts.entry(r.subreddit.clone()).or_default() += r.count;
        if progress_count % PB_INC == 0 { pb.inc(PB_INC) }
        progress_count += 1;
    }
    pb.finish();

    // Remove any subs below the comment threshold.
    records.retain(|record| counts[&record.subreddit[..]] > subreddit_comment_threshold);
    records
}

fn compute_overlaps_arr(author_vec: &Vec<HashMap<usize, i32>>, sub_total_comments: &HashMap<usize, i32>) -> Vec<Vec<f32>>{
    // First, we make an index:

    println!("Computing overlaps");
    let nr_subs = sub_total_comments.len();
    let mut overlaps = Array2::<f32>::zeros((nr_subs, nr_subs));
    let mut author_counter = 0;
    let total_nr_authors = author_vec.len() as f32;

    let pb = ProgressBar::new(total_nr_authors as u64);

    for subreddit_comments in author_vec {

        let total_comments_by_author:f32 = subreddit_comments.values().sum::<i32>() as f32;

        for (&sub1, &nr_comments_1) in subreddit_comments {
            for (&sub2, &nr_comments_2) in subreddit_comments {
                if sub1 != sub2 {
                    overlaps[[sub1, sub2]] += (nr_comments_1 * nr_comments_2) as f32 / (total_comments_by_author * sub_total_comments[&sub2] as f32);
                }
            }
        }
        author_counter += 1;
        if author_counter % PB_INC == 0 { pb.inc(PB_INC); }
    }
    pb.finish();

    let mut overlaps_vec = vec![vec![0.; nr_subs]; nr_subs];

    for i in 0..nr_subs {
        for j in 0..nr_subs {
            // Make symmetric.
            overlaps_vec[i][j] = overlaps[[j, i]].min(overlaps[[i, j]]);
            if overlaps_vec[i][j] > 0.5 {
                println!("sub ids: {} {}, overlap: {} nr_subs: {} {}", i, j, overlaps_vec[i][j], sub_total_comments[&i], sub_total_comments[&j])
            }
        }
    }

    println!("finished generating network matrix. Passing back to python.");
    overlaps_vec
}

fn make_author_vec(authors: &Vec<&str>, subreddits: &Vec<usize>, counts: &Vec<i32>) -> (Vec<HashMap<usize, i32>>, HashMap<usize, i32>) {

    // Keep track of the total number of comments in a subreddit.
    let mut sub_total_comments = HashMap::<usize, i32>::new();

    let mut author_dict = HashMap::<&str, HashMap<usize, i32>>::new();

    let pb = ProgressBar::new(authors.len() as u64);
    println!("Grouping by authors...");
    for i in 0..authors.len() {
        *author_dict.entry(authors[i]).or_default().entry(subreddits[i]).or_default() += counts[i];
        *sub_total_comments.entry(subreddits[i]).or_default() += counts[i];

        if i % PB_INC as usize == 0 { pb.inc(PB_INC) };
    }
    pb.finish();

    let author_vec: Vec<HashMap<usize, i32>> = author_dict
        .into_values()
        .collect();
    (author_vec, sub_total_comments)
}

fn assign_sub_ids(records: &Vec<DataRecord>) -> HashMap<String, usize> {
    records
        .into_iter()
        .map(|record| record.subreddit.clone())
        .unique()
        .enumerate()
        .map(|(id, sub)| (sub, id))
        .collect::<HashMap<String, usize>>()
}

fn build_adjacency_matrix(data_path: &str, min_count_bot_exclusion: i32, subreddit_comment_threshold: i32) -> (Vec<Vec<f32>>, HashMap<String, usize>){
    let mut records = load_records(data_path).unwrap();
    filter_records(&mut records, min_count_bot_exclusion, subreddit_comment_threshold);

    let id_to_sub = assign_sub_ids(&records);

    let authors = records.iter().map(|r| &r.author[..]).collect_vec();
    let subreddits = records.iter().map(|r| id_to_sub[&r.subreddit]).collect_vec();
    let counts = records.iter().map(|r| r.count).collect_vec();

    let (author_vec, sub_comment_counter) = make_author_vec(&authors, &subreddits, &counts);
    let overlaps = compute_overlaps_arr(&author_vec, &sub_comment_counter);

    (overlaps, id_to_sub)
}

fn build_edge_list(data_path: &str, min_count_bot_exclusion: i32, subreddit_comment_threshold: i32) -> (Vec<(usize, usize, f32)>, HashMap<String, usize>) {
    let (overlaps, id_to_sub) = build_adjacency_matrix(data_path, min_count_bot_exclusion, subreddit_comment_threshold);

    let mut edge_list = Vec::new();
    for (i, row) in overlaps.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            if val != 0. {
                edge_list.push((i, j, val));
            }
        }
    }

    (edge_list, id_to_sub)
}

#[pyfunction]
fn build_and_save_adjacency_matrix(
    data_path: &str,
    result_path: &str,
    sub_to_id_path: &str,
    min_count_bot_exclusion: i32,
    subreddit_comment_threshold: i32)
{
    let (overlaps, sub_to_id) = build_adjacency_matrix(data_path, min_count_bot_exclusion, subreddit_comment_threshold);
    save_adjacency_matrix(&overlaps, result_path);
    save_sub_to_id(&sub_to_id, sub_to_id_path);
}

#[pyfunction]
fn build_and_save_edge_list(
    data_path: &str,
    result_path: &str,
    sub_to_id_path: &str,
    min_count_bot_exclusion: i32,
    subreddit_comment_threshold: i32) 
{
    let (edge_list, sub_to_id) = build_edge_list(data_path, min_count_bot_exclusion, subreddit_comment_threshold);
    save_edge_list(&edge_list, result_path);
    save_sub_to_id(&sub_to_id, sub_to_id_path);
}

fn save_edge_list(data: &Vec<(usize, usize, f32)>, path: &str) {
    let mut wtr = Writer::from_path(path).unwrap();

    // Header.
    wtr.write_record(&["node_id_1", "node_id_2", "weight"]).unwrap();
    // Data.
    for record in data {
        wtr.serialize(record).unwrap();
    }
}

fn save_adjacency_matrix(data: &Vec<Vec<f32>>, path: &str) {
    let mut wtr = Writer::from_path(path).unwrap();

    for row in data {
        wtr.serialize(row).unwrap();
    }
}

fn save_sub_to_id(data: &HashMap<String, usize>, path: &str) {
    let mut wtr = Writer::from_path(path).unwrap();

    wtr.write_record(&["subreddit", "id"]).unwrap();

    for row in data {
        wtr.serialize(row).unwrap();
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn network_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_save_adjacency_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(build_and_save_edge_list, m)?)?;
    Ok(())
}
