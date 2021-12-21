use pyo3::prelude::*;
use ndarray::Array2;
use std::collections::HashMap;
use std::fs;
use itertools::Itertools;
use serde::Deserialize;

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "f0_")]
    count: i32,
    subreddit: String,
    author: String,
}

fn load_records(data_path: &str) -> Result<Vec<Record>, csv::Error> {

    let mut all_records = Vec::new();

    let paths = fs::read_dir(data_path).unwrap();

    for path in paths {

        let path = path.unwrap().path();
        println!("Reading file: {}", path.as_os_str().to_str().unwrap());
        let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_path(path).unwrap();

        for record in rdr.deserialize() {
            all_records.push(record?);
        }
    }

    return Ok(all_records)
}

fn is_valid_record(record: &Record, min_count_bot_exclusion: i32) -> bool {
    record.author == "[deleted]"
    || record.author == "AutoModerator"
    || (record.count > min_count_bot_exclusion && record.author.to_lowercase().ends_with("bot"))
}

fn filter_records(records: &mut Vec<Record>, min_count_bot_exclusion: i32, subreddit_comment_threshold: i32) -> &mut Vec<Record> {

    // Exclude deleted accounts and bots.
    records.retain(|record| is_valid_record(record, min_count_bot_exclusion));
    
    // Count how many comments any given subreddit has.
    let counts: HashMap<String, i32> = records
        .iter()
        .group_by(|record| record.subreddit.clone())
        .into_iter()
        .map(|(key, group)| (key, group.map(|r| r.count).sum()))
        .collect::<HashMap<String, i32>>();
    
    // Remove any subs below the comment threshold.
    records.retain(|record| counts[&record.subreddit] > subreddit_comment_threshold);
    
    records
}

fn compute_overlaps_arr(author_vec: &Vec<HashMap<usize, i32>>, sub_total_comments: &HashMap<usize, i32>) -> Vec<Vec<f32>>{
    // First, we make an index:

    let nr_subs = sub_total_comments.len();
    let mut overlaps = Array2::<f32>::zeros((nr_subs, nr_subs));
    let mut author_counter = 0;

    let total_nr_authors = author_vec.len() as f32;

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
        if author_counter % 500000 == 0 {
            println!("nr authors: {} {} % done", author_counter, 100. * (author_counter as f32) / total_nr_authors);
        }
    }

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
    for i in 0..authors.len() {
        *author_dict.entry(authors[i]).or_default().entry(subreddits[i]).or_default() += counts[i];
        *sub_total_comments.entry(subreddits[i]).or_default() += counts[i];
    }

    let author_vec: Vec<HashMap<usize, i32>> = author_dict
        .into_values()
        .collect();
    (author_vec, sub_total_comments)
}

fn assign_sub_ids(records: &Vec<Record>) -> HashMap<String, usize> {
    records
        .into_iter()
        .map(|record| record.subreddit.clone())
        .unique()
        .enumerate()
        .map(|(id, sub)| (sub, id))
        .collect::<HashMap<String, usize>>()
}

#[pyfunction]
fn build_network_from_raw(data_path: &str, min_count_bot_exclusion: i32, subreddit_comment_threshold: i32) -> (Vec<Vec<f32>>, HashMap<String, usize>) {
    let mut records = load_records(data_path).unwrap();
    filter_records(&mut records, min_count_bot_exclusion, subreddit_comment_threshold);
    let id_to_sub = assign_sub_ids(&records);

    let authors = records.iter().map(|r| &r.author[..]).collect_vec();
    let subreddits = records.iter().map(|r| id_to_sub[&r.subreddit]).collect_vec();
    let counts = records.iter().map(|r| r.count).collect_vec();

    let (author_vec, sub_comment_counter) = make_author_vec(&authors, &subreddits, &counts);
    let overlaps = compute_overlaps_arr(&author_vec, &sub_comment_counter);

    return (overlaps, id_to_sub);
}

#[pyfunction]
fn build_network(authors: Vec<&str>, subreddits: Vec<usize>, counts: Vec<i32>) -> Vec<Vec<f32>> {
    let (author_vec, sub_total_comments) = make_author_vec(&authors, &subreddits, &counts);
    compute_overlaps_arr(&author_vec, &sub_total_comments)
}


/// A Python module implemented in Rust.
#[pymodule]
fn network_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_network, m)?)?;
    m.add_function(wrap_pyfunction!(build_network_from_raw, m)?)?;
    Ok(())
}
