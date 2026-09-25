# AmazonML26 Data Layer

## Overview

The data layer uses Polars and Parquet for efficient processing of the large
training datasets.

The original competition data is provided as TSV files. These are converted
once into Parquet files and then accessed through `src/database.py`.

---

## Data Flow

```text
Original TSV
     |
     v
   Polars
     |
     v
  Parquet
     |
     v
database.py
     |
     +-- scan()
     +-- read()
     +-- count()
     +-- get_columns()
     +-- filter_by_country()
     +-- select_columns()
     +-- get_by_id()
     +-- get_by_ids()
     +-- get_matches()