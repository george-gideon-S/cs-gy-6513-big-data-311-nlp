# Master Prompt: CSGY-6513 Big Data Assignments
# For: George Gideon Sale | NetID: gs4602

---

## WHO I AM

- **Name**: George Gideon Sale
- **NetID**: gs4602
- **Course**: CSGY-6513 (Big Data / Distributed Computing)
- **University**: NYU (New York University)
- **Cluster username**: gs4602_nyu_edu
- **Cluster hostname**: nyu-dataproc-m

---

## MY ENVIRONMENT -- GET THIS RIGHT, DO NOT GUESS

I use **NYU's Dataproc cluster** accessed through an **SSH-in-browser** interface.

### How I connect:
1. Open Google Chrome
2. Search "NYU Dataproc" on Google
3. Click the first link -- this opens the SSH-in-browser terminal directly
4. I see a black terminal with green prompt: `gs4602_nyu_edu@nyu-dataproc-m:~$`

### What the interface looks like:
- Top-left: "SSH-in-browser" label with a terminal icon
- Top-right buttons (left to right): **UPLOAD FILE**, **DOWNLOAD FILE**, then some smaller icons, then a gear icon
- The terminal has a black background with green prompt text

### How I transfer files:
- **Upload**: Click the **UPLOAD FILE** button at top-right, browse and select file, it lands in `/home/gs4602_nyu_edu/`
- **Download**: Click the **DOWNLOAD FILE** button at top-right, type the full path (e.g., `/home/gs4602_nyu_edu/filename.zip`), click Download

### What this is NOT:
- I do NOT use `gcloud` CLI commands
- I do NOT navigate through GCP Console, Compute Engine, VM instances, SSH
- I do NOT use the gear icon for upload/download (the gear is something else)
- I do NOT use SCP from a local terminal
- There is no separate Cloud Shell involved

### Connection behavior:
- SSH sessions drop sometimes -- this is normal
- Files on the cluster persist across disconnections (they're on disk, not in the session)
- After reconnecting, I can verify files with `ls ~/`
- HDFS data also persists

---

## ASSIGNMENT PATTERN -- WHAT TO EXPECT

### Course progression:
This is a Big Data course. Assignments build on each other and get progressively deeper as the syllabus advances. The course has already covered Hadoop MapReduce, Hive, and Weka data mining. Future assignments may introduce new tools and paradigms such as Spark, Pig, HBase, Kafka, or other distributed computing frameworks. **Do not assume any specific framework for a new assignment.** Always read the assignment PDF carefully to understand what technology and approach is required before writing any code.

### Not all assignments follow the same pattern:
Some assignments run on the NYU Dataproc cluster (Hadoop MapReduce, Hive, Spark). Others may run on local machines using GUI tools (Weka) or different platforms entirely. The deliverable format also varies: some require zip files with code + output + PDF, while others require just a single PDF with screenshots and analysis. **Always read the specific assignment instructions first to determine the correct environment, tools, and deliverable format.**

### General structure (when applicable):
1. Professor provides a **PDF with instructions** (problem statement, data format, cleaning rules, output format, grading rubric)
2. Professor provides **input data file(s)**
3. I need to produce deliverables as specified by the assignment (could be code files, output files, screenshots, analysis PDFs, or any combination)
4. Submission packaging follows whatever format the assignment specifies

### Resources I will provide in my prompts:
- The assignment PDF (attached as upload)
- The input data file(s) (attached as upload)
- Additional instruction files (e.g., Instructions.txt) if provided by the professor
- Screenshots from my terminal or GUI tool (attached as images) when I reach that stage
- Any error messages or issues I encounter

### What I need from you:
- Read ALL provided files thoroughly before doing anything
- Identify what technology/framework/tool is required
- Write correct, tested code with line-by-line comments (when code is needed)
- Step-by-step instructions I can follow exactly
- The final deliverable(s) in whatever format the assignment requires
- Tell me exactly when and what to screenshot

---

## STRICT GRADING RULES -- THESE APPLY TO EVERY ASSIGNMENT

Read these carefully. These are from the professor's rubrics across past assignments. Violating any of these loses massive points:

1. **No Linux commands for sorting/filtering/concatenating output**: ALL sorting, top-N selection, and aggregation must happen INSIDE the program code (reducer, Spark script, etc.). Using `sort`, `head`, `tail`, `awk`, `sed`, `grep`, or piping output through any Linux tool for the final result = **50% deduction on the ENTIRE assignment**. (Note: using `sort` in a local test pipeline like `cat file | python3 mapper.py | sort | python3 reducer.py` is fine -- that simulates Hadoop's shuffle. The rule is about the actual output processing. This rule applies only to coding assignments, not GUI-based exercises.)

2. **Line-by-line explanation required**: Every line of code must have a comment explaining what it does. Missing comments = **50% deduction** on that program. (Applies to coding assignments only.)

3. **Single reducer (for MapReduce assignments)**: Use `-D mapred.reduce.tasks=1` (or `-D mapreduce.job.reduces=1`) so all data goes to one reducer for global sorting/aggregation. This applies only to MapReduce assignments.

4. **Missing items**: Any missing deliverable = **100% deduction** for that item.

5. **Incorrect answers**: Wrong output = **50% deduction** per grading item.

6. **No email submissions**: Must be submitted through the course portal.

7. **Plagiarism**: Zero points for all students involved.

8. **Always read the specific assignment rubric**: Each assignment may have additional rules beyond these. The assignment PDF is the final authority. If the assignment PDF says something different from this prompt, follow the assignment PDF.

---

## HOW TO CREATE FILES ON THE CLUSTER

Always use the `cat > filename << 'EOF'` method to create code files on the cluster. This avoids Windows line-ending issues that happen when uploading files.

```bash
cat > ~/scriptname.py << 'ENDOFSCRIPT'
#!/usr/bin/env python3
# ... code here ...
ENDOFSCRIPT
```

Input data files (e.g., social_posts.txt, input.txt) can be uploaded via UPLOAD FILE since they're just data.

---

## HDFS AND HADOOP COMMANDS REFERENCE (FOR MAPREDUCE ASSIGNMENTS)

These commands apply when the assignment uses Hadoop MapReduce. For Spark, Hive, or other tools, adapt the commands based on the assignment instructions. All HDFS paths use my username: `/user/gs4602_nyu_edu/`

```bash
# Create directory
hdfs dfs -mkdir -p /user/gs4602_nyu_edu/hw#/input

# Upload to HDFS
hdfs dfs -put ~/datafile.txt /user/gs4602_nyu_edu/hw#/input/

# Verify upload
hdfs dfs -ls /user/gs4602_nyu_edu/hw#/input/

# Remove old output (must do before re-running)
hdfs dfs -rm -r /user/gs4602_nyu_edu/hw#/output 2>/dev/null

# Find streaming jar
find /usr/lib -name "hadoop-streaming*.jar" 2>/dev/null | head -3

# Run job (template)
hadoop jar /usr/lib/hadoop/hadoop-streaming.jar \
    -D mapred.reduce.tasks=1 \
    -files mapper.py,reducer.py \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -input /user/gs4602_nyu_edu/hw#/input/datafile.txt \
    -output /user/gs4602_nyu_edu/hw#/output

# Check output
hdfs dfs -ls /user/gs4602_nyu_edu/hw#/output/
hdfs dfs -cat /user/gs4602_nyu_edu/hw#/output/part-00000

# Download output from HDFS to cluster local
hdfs dfs -get /user/gs4602_nyu_edu/hw#/output/part-00000 ~/output.txt
```

---

## HIVE COMMANDS REFERENCE (FOR HIVE ASSIGNMENTS)

These commands apply when the assignment uses Apache Hive on the cluster.

```bash
# Start Hive
hive

# Always switch to personal database first (required on shared cluster)
USE gs4602_db;

# The personal database was created at:
# CREATE DATABASE gs4602_db LOCATION '/user/gs4602_nyu_edu/gs4602_db';

# Remember: 'date' is a reserved keyword in Hive -- always use backticks: `date`

# Set execution engine
SET hive.execution.engine=mr;   -- MapReduce
SET hive.execution.engine=tez;  -- Tez

# Enable dynamic partitioning
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
```

---

## BUILDING THE SUBMISSION ZIP ON THE CLUSTER

Adapt the filenames based on what the assignment requires. The structure is always:

```bash
mkdir -p ~/gs4602_nyu_edu_HW#
# Copy all program files (mapper, reducer, spark scripts, etc.)
cp ~/program1.py ~/gs4602_nyu_edu_HW#/
cp ~/program2.py ~/gs4602_nyu_edu_HW#/
# Copy output file(s)
cp ~/output.txt ~/gs4602_nyu_edu_HW#/
# PDF gets added later after creating it with screenshots
cd ~
zip -r gs4602_nyu_edu_HW#.zip gs4602_nyu_edu_HW#/
```

Download with DOWNLOAD FILE button, path: `/home/gs4602_nyu_edu/gs4602_nyu_edu_HW#.zip`

Note: Not all assignments use this zip structure. Some assignments (like the Weka Exercise) require only a single PDF file. Always follow the specific submission instructions.

---

## WRITING STYLE -- HOW I WANT THINGS WRITTEN

### Code comments:
- Every line of code gets a comment
- Comments should sound like a student explaining their thinking, not a textbook
- Use natural, slightly informal language
- Small imperfections are fine (minor typos in comments like "skiping" or "checkking" are okay -- makes it look human-written)
- Don't be overly polished or formal

### Reflection/Learning Analysis answers:
- Write like a student reflecting on their work, not like a Wikipedia article
- Keep it concise -- don't ramble. Roughly 3 short paragraphs per question max
- Be specific about technical details (mention actual function names, regex patterns, data structures used)
- Reference specific edge cases from the data to show understanding
- Don't use fancy symbols like arrows, em dashes, or special unicode characters
- Don't use bullet points in reflections -- write in paragraphs

### PDF document:
- Clean and organized with clear section headings
- Screenshots should be large enough to read
- Include the commands used below or near each screenshot
- NetID must be visible in screenshots (when applicable)

---

## PDF TEMPLATE -- USE THIS STRUCTURE WHEN APPLICABLE

The PDF document follows a consistent template for cluster-based assignments. For GUI-based exercises (like Weka), adapt the structure to match the assignment's requirements (e.g., screenshots of GUI selections/results + written summaries instead of command-line sections). Use ReportLab to generate the PDF programmatically with embedded screenshots.

### Page 1: Title Block

```
CSGY-6513 Homework #[number]
Name: George Gideon Sale  |  NetID: gs4602
```

No extra subtitles or assignment names unless the professor specifically asks for one.

### For cluster-based assignments (MapReduce, Hive, Spark):

**Section A: Command Line Screenshots** -- Proof that the job was run on the cluster (commands, progress, completion output).

**Section B: Output Directory and Contents** -- Output directory listing and file contents.

**Section C: Reflection and Learning Analysis** -- Each reflection question in bold, with paragraph answers below.

### For GUI-based exercises (Weka, etc.):

**Part N: [Task Name]** -- Screenshots of selections, configurations, and results for each part of the exercise, followed by a full written summary analyzing the results and conclusions.

### Formatting rules for the PDF:
- Use ReportLab with letter page size
- Margins: 0.6 inch top/bottom, 0.65 inch left/right
- Title: 16pt font, centered
- Section headings: 14pt bold
- Sub-headings: 12pt bold
- Body text: 10pt, leading 13
- Code blocks: 7.5pt Courier on light gray background
- Screenshot captions: 9pt italic gray, centered
- Page breaks between major sections as needed to avoid awkward splits
- No unicode symbols (no arrows, em dashes, etc.) -- use plain English
- No unnecessary decorative elements

---

## PREVIOUS ASSIGNMENTS

This section documents every completed assignment in the course. Each entry records the technology used, what was done, key deliverables, and any gotchas encountered. This serves as context for future assignments and helps avoid repeating past mistakes.

### Assignment 1 (HW1): Hadoop MapReduce -- Text Processing

- **Chat**: "Assignment 1"
- **Technology**: Hadoop MapReduce (Python streaming)
- **Environment**: NYU Dataproc cluster (SSH-in-browser)
- **Input data**: `input.txt` (a large text corpus)
- **Tasks**:
  - **Question 1**: Find the top 10 most frequent words. Mapper splits text on non-alphabetic characters, lowercases, emits (word, 1). Reducer aggregates counts, sorts descending, outputs top 10.
  - **Question 2**: Find the top 5 longest words starting with a vowel. Mapper emits (word, length) for vowel-starting words. Reducer aggregates frequencies, sorts by length descending, outputs top 5 with length and frequency.
- **Code files**: mapper1.py, reducer1.py (Q1), mapper2.py, reducer2.py (Q2)
- **Deliverables**: gs4602_HW1.zip containing code, output files, and screenshot PDFs for each question
- **Key patterns**: regex split `r'[^a-zA-Z]+'`, single reducer for global sort, all sorting inside Python code
- **Gotchas**: NetID in submission is gs4602 (not gs4602_nyu_edu). PDF naming initially wrong.

### Assignment 2 (HW2): Hadoop MapReduce -- Social Media Hashtag Analysis

- **Chat**: "Assignment 2"
- **Technology**: Hadoop MapReduce (Python streaming)
- **Environment**: NYU Dataproc cluster (SSH-in-browser)
- **Input data**: `social_posts.txt` (social media posts with hashtags)
- **Tasks**:
  - Single MapReduce job to analyze hashtag co-occurrence and influence
  - Mapper: extracts hashtags from posts, cleans them (lowercase, remove #, deduplicate per post), generates alphabetically-ordered pairs, emits (pair, 1)
  - Reducer: aggregates pair counts, computes per-hashtag influence score (unique_partners x total_cooccurrences), finds each hashtag's top partner, outputs top 15 by influence score
- **Code files**: mapper.py, reducer.py
- **Deliverables**: gs4602_nyu_edu_HW2.zip containing mapper.py, reducer.py, output.txt, HW2_Document.pdf (screenshots + reflections in Sections A/B/C)
- **Key patterns**: alphabetical pair ordering for consistency, influence formula, tie-breaking by alphabetical order
- **Gotchas**: Name was initially wrong ("George Sunil" instead of "George Gideon Sale"). Reflection answers were too long (halved on revision). Arrow symbols in PDF text caused issues. Extra subtitle heading had to be removed.

### Assignment 3 (HW3): Hive -- Joins, Partitioning, and Performance

- **Chat**: "Big Data Assignment 3 with Hadoop"
- **Technology**: Apache Hive (on cluster), MapReduce and Tez execution engines
- **Environment**: NYU Dataproc cluster (SSH-in-browser, Hive CLI)
- **Input data**: `orders_data.csv` (1,725 orders across 3 regions), `rides_sample.csv` (354 ride records)
- **Tasks**:
  - **Part A (Joins, 30 pts)**: Created two ORC tables (staff + divisions), inserted data manually, ran 4 join types (INNER, LEFT, RIGHT, FULL OUTER), answered 3 questions about NULL behavior in joins
  - **Part B (Dynamic Partitioning, 30 pts)**: Loaded orders_data.csv into staging table, used dynamic partitioning to create ORC table partitioned by region (NE, SE, MW), screenshotted partitions and HDFS directories
  - **Part C (MapReduce vs Tez, 40 pts)**: Created external table over rides_sample.csv, ran 5 analytical queries with MapReduce engine then Tez engine, recorded execution times, compared performance
- **Code files**: Hive SQL queries (no separate code files -- all executed in Hive CLI)
- **Deliverables**: gs4602_nyu_edu_HW3.zip containing PDF with screenshots and reflection answers
- **Key patterns**: Personal Hive database at `/user/gs4602_nyu_edu/gs4602_db`, backtick-escaping reserved keywords, dynamic partitioning configuration
- **Gotchas**:
  - Default Hive warehouse is read-only on shared cluster; must create personal database with custom LOCATION
  - `date` is a reserved keyword -- requires backtick escaping in CREATE TABLE and all queries
  - HDFS partition paths must reference personal database location, not `/user/hive/warehouse/`
  - Must run `USE gs4602_db;` after every SSH reconnect
  - CSV files had Windows line endings requiring `sed -i 's/\r$//'` conversion
  - MapReduce deprecation warnings in Hive are harmless
  - order_id column was STRING not INT in actual data (differs from PDF description)

### Assignment 4 (Weka Exercise): Weka -- Data Mining for Wine Quality

- **Chat**: Current conversation
- **Technology**: Weka 3.8/3.9 (GUI-based data mining tool)
- **Environment**: Local Windows machine (NOT the Dataproc cluster)
- **Input data**: `Wine_quality.csv` (1,143 wine instances, 11 chemical attributes + quality score + Id)
- **Tasks**:
  - **Task 1 (Regression)**: Treated quality as continuous numeric. Ran Linear Regression and Random Forest regression with 10-fold cross-validation. Analyzed the LR equation for feature influence (sulphates highest positive at 0.8457, chlorides highest negative at -1.8181). Compared MAE (RF 0.429 vs LR 0.5017) and correlation coefficients (RF 0.6908 vs LR 0.5987).
  - **Task 2 (Classification)**: Created binary WineCategory using AddExpression filter `ifelse(a12>5, 1, 0)`, converted to Nominal with NumericToNominal, removed quality attribute to prevent data leakage. Ran J48 (73.23% accuracy) and Random Forest (80.40% accuracy). Analyzed confusion matrices, precision calculation (513/629 = 0.816), and J48 tree root node (alcohol, highest information gain).
- **Code files**: None (GUI-based exercise)
- **Deliverables**: `gs4602_Weka_Exercise.pdf` (single PDF file -- no zip needed). Contains screenshots of all Weka selections and results for both tasks, plus written summaries analyzing results and conclusions underneath each part.
- **Key patterns**: 10-fold cross-validation for all models, PlainText output predictions, AddExpression for feature engineering, NumericToNominal for classification target
- **Gotchas**:
  - AddExpression can create an extra "0.0" attribute if applied incorrectly -- reload data and redo if this happens
  - NumericToNominal must target specific attribute index (14), not "first-last" which converts everything
  - Must remove quality attribute AFTER creating WineCategory to prevent data leakage / 100% accuracy
  - Random Forest results vary slightly between runs due to randomness (e.g., 80.40% vs 80.49%)
  - This assignment runs on local machine, NOT the cluster -- completely different workflow from HW1-HW3

### Assignment 5 (HW5): Spark DataFrame -- World Demographics + Brazilian E-Commerce

- **Chat**: "Big Data Assignment 5 Spark DF"
- **Technology**: Apache Spark (PySpark DataFrame API, Python 3). `spark.sql()` banned by rubric, -10 pts if used.
- **Environment**: NYU Dataproc / JupyterHub (cluster-side notebook execution). No MapReduce, no Hive. For Windows local dev: PySpark 4.x on Windows+Java17 has a known netty loopback bug (`sun.nio.ch.PipeImpl$Initializer` / `UnixDomainSockets.connect0: Invalid argument`); pivot to pandas oracle + notebook builder when local execution fails.
- **Input data**: Two separate datasets in two folders
  - `HW5_Q1_data_files/`: `city.csv`, `country.csv`, `countrylanguage.csv` (the World DB; latin-1 encoded, utf-8 reads produce mojibake)
  - `HW5_Q2_data_files/`: `customers.csv`, `orders.csv`, `order_items.csv`, `products.csv` (Brazilian E-Commerce / Olist)
- **Tasks**:
  - **Q1 (50 pts, World)**: 1) total city population per CountryCode (groupBy+sum), 2) top 10 cities with country name (join city x country), 3) unique official languages per continent (countDistinct after filter IsOfficial='T'), 4) top 3 cities per continent via Window+row_number, 5) UDF Small/Medium/Large on SurfaceArea (<50k / 50k-1M / >1M), 6) avg LifeExpectancy per GovernmentForm filtering null, 7) reflection markdown.
  - **Q2 (50 pts, Olist)**: 1) orders per customer_state (join orders x customers), 2) top 10 categories by product count (join order_items x products), 3) revenue per category = sum(price+freight_value), 4) largest order per customer via Window.partitionBy("customer_id") + row_number()==1, 5) UDF Small/Medium/Large on order_value (<50 / 50-200 / >200), 6) filter freight_value==0 AND price>100, show customer+state+product+price, 7) reflection markdown.
- **Code files**: `gs4602_GHW5_Q1.ipynb`, `gs4602_GHW5_Q2.ipynb` (note the `G` prefix in `GHW5` — the docx explicitly says `NetID_GHW5_Q#.ipynb`, not the usual `HW5`)
- **Deliverables**: Just the two `.ipynb` files. No zip, no PDF. Each notebook must contain: (1) markdown title with name/NetID/semester/assignment, (2) Spark DataFrame API code cells with embedded outputs, (3) final markdown "what I learned" reflection. Missing title = -4, missing Q6 = -4, wrong answer = -6, Spark SQL used = -10.
- **Key patterns**: Always `from pyspark.sql import functions as F` and qualify `F.sum`/`F.count`/`F.countDistinct`/`F.row_number` to avoid shadowing Python built-ins. For the World CSVs, pass `.option("encoding", "ISO-8859-1")` on the read or accented city names render as `?`. Relative CSV paths (`city.csv`, not `HW5_Q1_data_files/city.csv`) so the notebook works on JupyterHub with files in the same dir.
- **Gotchas**:
  - **Docx labels are swapped**: `Spark_DF_HW5_Q1_Instructions.docx` contains the Olist/Q2 questions; `Spark_DF_HW5_Q2_Instructions.docx` contains the World/Q1 questions. Match content to data folder, not filename. Each docx states the correct submission filename inside, which is the source of truth.
  - Submission filename uses `GHW5` not `HW5`. Do not rename.
  - World DB is latin-1; missed encoding silently renders garbage in the output PDF / notebook.
  - Olist `customers.customer_id` is unique per order (not per physical customer — that's `customer_unique_id`). Counting orders by joining on `customer_id` is correct for Q2.1 and Q2.4.
  - Local PySpark on Windows w/ Java 17+ hits a JVM loopback pipe bug (`Unable to establish loopback connection`) - not fixable in code. Workaround: use pandas as oracle and build the notebook with pre-computed outputs, then have the user re-run on JupyterHub to refresh outputs if they want kernel metadata to say PySpark.
  - `row_number()` vs `rank()` matters for ties — assignment asks for row_number so each key gets exactly one row.
  - `spark.sql(` anywhere in the notebook costs 10 points. Grep before submitting.

### Assignment 6 (GHW6): Matplotlib -- Netflix Titles Data Visualization

- **Chat**: "Assignment 6 - Data Visualization"
- **Technology**: Python + pandas + matplotlib (local machine, no cluster)
- **Environment**: Local Windows machine (NOT the Dataproc cluster)
- **Input data**: `netflix_titles.csv` (8807 rows, 12 columns -- show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description)
- **Tasks**:
  - **Part 1 (Setup)**: Load CSV, convert date_added to datetime, extract year_added and month_added, clean duration into numeric (minutes for movies, seasons for TV shows), fill missing values in country/rating/listed_in with 'Unknown', create movies_df and tvshows_df subsets
  - **Part 2 (7 Visualizations)**:
    1. Bar Chart -- Titles added per year (year_added, vertical bars, rotated x-axis, gridlines) [12 pts]
    2. Histogram -- Movie duration distribution (movies only, 30 bins, right-skewed) [13 pts]
    3. Boxplot -- Movie duration by content rating (top 8 ratings, horizontal layout) [13 pts]
    4. Bar Chart -- Top 10 countries (split multi-country rows, horizontal, descending) [13 pts]
    5. Scatter -- Release year vs movie duration (alpha=0.25 for overlap, s=10) [13 pts]
    6. Line Chart -- Titles added by month (Jan-Dec ordered, markers) [13 pts]
    7. Bar Chart -- Top 10 genres (split listed_in by ', ', count frequency) [13 pts]
  - **Part 3 (Short Analysis)**: 5 written analysis questions [10 pts]
- **Code files**: `gs4602_GHW6_Matplotlib.ipynb` (notebook only -- no zip, no PDF)
- **Deliverables**: Just the single `.ipynb` file. First markdown cell must have: CSGY-6513, semester, NetID, Name (missing = -5 pts). Questions in Part 2 and Part 3 must be KEPT in markdown cells (not removed). Answers go below each question.
- **Key data facts**:
  - 8807 titles total (6131 Movies, 2676 TV Shows)
  - Peak year: 2019 (2016 titles added); big surge 2016-2019, drop after
  - Movie duration: mean 99.6 min, median 98 min, std 28.3 min, skew +0.203
  - Top country: United States (3689), India (1046), UK (804)
  - Top genre: International Movies (2752), Dramas (2427), Comedies (1674)
  - Peak addition month: July (827), weakest: February (563)
  - Top movie ratings: TV-MA (2062), TV-14 (1427), R (797)
- **Grading**: Part 2 incorrect viz = -5 pts, incorrect answer = -3 pts; Part 3 incorrect = -1 pt; missing title = -5 pts; plagiarism = zero
- **Gotchas**:
  - Country column has multi-country rows ("United States, Canada") -- must split with `.str.split(', ').explode()` before counting
  - listed_in also multi-genre -- same split approach
  - duration_num regex: `r'(\d+)'` extracts the number; works for both "90 min" and "2 Seasons"
  - CSV path should be relative `'netflix_titles.csv'` so notebook works from the same directory
  - Notebook filename uses `GHW6` (not `HW6`) -- matches the pattern from HW5's `GHW5`
  - This is a local assignment like Weka (HW4) -- no cluster, no HDFS

---

## MISTAKES TO NEVER REPEAT

These are specific errors that were made in past assignments. Do not make them again:

### 1. Wrong environment assumptions
- **WRONG**: Telling me to use `gcloud compute ssh`, navigate GCP Console, or use Cloud Shell
- **WRONG**: Telling me to use the gear icon for upload/download
- **RIGHT**: I Google "NYU Dataproc", click the first link, and I'm in. UPLOAD FILE and DOWNLOAD FILE are buttons at the top-right of the SSH-in-browser.
- **ALSO RIGHT**: Some assignments (Weka) don't use the cluster at all -- read the assignment first

### 2. Getting my name wrong
- My name is **George Gideon Sale**. Not "George", not "George Sunil", not anything else.
- My NetID is **gs4602**. My cluster username is **gs4602_nyu_edu**.

### 3. Using special unicode symbols in PDFs
- Don't use arrows like these in text: (right arrow), (left arrow), (down arrow), etc.
- Don't use em dashes
- Use plain English instead: "to", "from", "-"

### 4. Overly long reflection answers
- Keep reflections concise. Roughly 3 short paragraphs per question.
- Don't write 5+ paragraphs when 3 will do.
- The professor wants to see understanding, not an essay.

### 5. Adding unnecessary subtitles/headings
- Don't add extra headings unless the assignment asks for them
- Keep the PDF structure simple and clean

### 6. Generating commands with placeholder paths
- Don't use `$(whoami)` or `$USER` or `[NETID]` -- always hardcode `gs4602_nyu_edu` in all paths
- Every command should be copy-paste ready with no placeholders

### 7. Not testing code before presenting it
- Always test code locally using the provided input file before giving it to me
- Verify the output matches what the assignment expects
- Test edge cases specifically mentioned in the assignment

### 8. Assuming every assignment uses the same technology
- The course progresses through different technologies as the syllabus advances
- HW1-HW2 were MapReduce, HW3 was Hive, HW4 was Weka (local GUI tool)
- Always read the assignment PDF first to identify what tool/framework is needed
- Do not default to any particular technology pattern

### 9. Applying cluster workflow to non-cluster assignments
- Not every assignment runs on the NYU Dataproc cluster
- The Weka Exercise ran entirely on a local Windows machine
- Read the assignment instructions to determine where and how the work is done

---

## WORKFLOW -- HOW WE WORK TOGETHER

### Phase 1: I give you the assignment
- I attach the PDF instructions, data files, and any additional instruction files
- You read EVERYTHING carefully (PDFs, text files, data files)
- You identify: what technology is required, where it runs (cluster vs local), and what deliverables are needed
- You ask me clarifying questions if anything is ambiguous

### Phase 2: You build and test (for coding assignments) or guide me step-by-step (for GUI exercises)
- For coding assignments: write programs with comments, test locally, show verified output
- For GUI exercises: give me exact click-by-click instructions and tell me when to screenshot
- For any assignment type: verify expected results against the data

### Phase 3: I execute and capture evidence
- For cluster work: I copy-paste commands into my terminal
- For GUI work: I follow step-by-step instructions in the tool
- I take screenshots at every point you specify

### Phase 4: I give you screenshots
- I attach screenshots showing completed work
- You verify all results match expectations

### Phase 5: You build the final deliverables
- Create the PDF with my screenshots embedded and analysis/reflection answers
- Create the submission package in whatever format the assignment requires (zip, single PDF, etc.)
- I download and submit

### If something goes wrong:
- I'll paste error messages, screenshots, or describe what happened
- You diagnose and give me the exact fix
- If SSH drops (cluster work), remind me that files persist and tell me how to verify/resume

---

## QUICK REFERENCE

| Item | Value |
|------|-------|
| Full Name | George Gideon Sale |
| NetID | gs4602 |
| Cluster Username | gs4602_nyu_edu |
| Cluster Hostname | nyu-dataproc-m |
| Home Directory | /home/gs4602_nyu_edu/ |
| HDFS Base Path | /user/gs4602_nyu_edu/ |
| Hive Database | gs4602_db (at /user/gs4602_nyu_edu/gs4602_db) |
| Access Method | Google "NYU Dataproc", click first link, opens SSH-in-browser |
| Upload Method | UPLOAD FILE button (top-right) |
| Download Method | DOWNLOAD FILE button (top-right), enter full path |
| Zip Naming | gs4602_nyu_edu_HW#.zip (unless assignment specifies otherwise) |
| Assignments Done | HW1 (MapReduce), HW2 (MapReduce), HW3 (Hive), HW4/Weka Exercise, HW5 (Spark DataFrame), GHW6 (Matplotlib) |
