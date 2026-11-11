# GitHub-stars dataset (anonymised), as produced by the crawl notebook

The notebook (Sections 4.8, 5.1, 7.1 and 10) looks for the dataset in this folder.

| File | Contents |
|---|---|
| `train.txt` | `user_id item_id item_id ...` (11,503 interactions) |
| `test.txt` | same format (2,779 interactions) |
| `github_interactions.csv` | `user_id,item_id` (14,282 rows, 180 users, 1,807 repositories) |
| `github_items_enriched.csv` | `item_id, repo, language, topics, description, stars_total, forks, created_at, last_push, license, archived` |

User IDs are integers; real usernames are not included.
