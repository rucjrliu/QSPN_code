# QSPN_code

This is the code of our *QSPN/M-QSPN* model, corresponding to the paper: A Unified Model for Cardinality Estimation by Learning from Data and Queries via Sum-Product Networks. In this paper, we introduces *QSPN*, a unified model that integrates both data distribution and query workload. *QSPN* achieves high estimation ac- curacy by modeling data distribution using the simple yet effective Sum-Product Network (SPN) structure. To ensure low inference time and reduce storage overhead, *QSPN* further partitions columns based on query access patterns. We formalize *QSPN* as a tree-based structure that extends SPNs by introducing two new node types: QProduct and QSplit. This paper studies the research challenges of developing efficient algorithms for the offline construction and online computation of *QSPN*. We conduct extensive experiments to evaluate *QSPN* in both single-table and multi-table cardinality estimation settings. The experimental results have demonstrated that *QSPN* achieves superior and robust performance on the three key criteria, compared with state-of-the-art approaches.

## Configuration

**Step 1:** Download this code and unzip it.

**Step 2:** Install dependencies of the environment qspn.

    cd QSPN_code/
    conda install -f environment.yml

**In the following, all running should be at this directory 'QSPN_code/'.**

**Step 3:** Downloading our experimental data.

Download file **qspn_data_models.tar** from OneDrive sharing link: https://1drv.ms/u/c/f9c0a1a8c6911768/EZhXKSN0fgVBizPkvw0Eeu0BNVAjk5OxrwoVerEp_ojylg?e=cErDPM and then decompress it:

    tar -xf qspn_data_models.tar

There will be two new directories in *'QSPN_code/'*: *'data/'* and *'models/'* which store datasets, workloads and models of our *QSPN/M-QSPN* experiment.

## Quick Start

### QSPN Single-Table CardEst

We first show the commands of conducting the experiments in our paper.

Dataset: GAS/Census/Forest/Power

    python scripts/run_qspn.py --dataset gas --inference
    python scripts/run_qspn.py --dataset census13 --inference
    python scripts/run_qspn.py --dataset forest10 --inference
    python scripts/run_qspn.py --dataset power7 --inference

The result will be outputed to the console, including Model Size, Mean Inference Time, Q-error of each query, 50th/90th/95th/99th/Max/Mean Q-error of CardEst on the workload.

You can also construct *QSPN* models by yourself, like

    python scripts/run_qspn.py --dataset gas --train
    python scripts/run_qspn.py --dataset census13 --train
    python scripts/run_qspn.py --dataset forest10 --train
    python scripts/run_qspn.py --dataset power7 --train

The Model Construction Time and Model Size will be outputed to the console. And the *QSPN* model file: *'models/single_tables/qspn/qspn_5.0_0.1_0.3_QSplit_\{gas/census13/forest10/power7\}_template_0.5_0.5.pkl'* will be overwritten by the new constructed *QSPN* model.

### QSPN Update CardEst

Take dataset Forest for instance, we show how to run three update methods: *NoTrain*, *ReTrain* and our *AdaIncr*.

    python scripts/run_qspn.py --dataset forest10 --skew 0.0 --corr 0.0 --update-method notrain --update-query-root update --update-skew 0.5 --update-corr 0.5 --update
    python scripts/run_qspn.py --dataset forest10 --skew 0.0 --corr 0.0 --update-method retrain --update-query-root update --update-skew 0.5 --update-corr 0.5 --update
    python scripts/run_qspn.py --dataset forest10 --skew 0.0 --corr 0.0 --update-method adaincr --update-query-root update --update-skew 0.5 --update-corr 0.5 --update

where '--skew 0.0' and '--corr 0.0' means using the model constructed on the original workload with no access on correlated columns; '--update-method notrain/retrain/adaincr' means testing different update method *(NoTrain/ReTrain/AdaIncr)*; '--update-query-root update --update-skew 0.5 --update-corr 0.5' means testing the model on Hybrid-Update workload ('--update-query-root update': the workload contains data-update; '--update-skew 0.5 --update-corr 0.5': the workload contains query-update).

If you want to test update method *AdaIncr* on Data-Update-Only workload, the command is

    python scripts/run_qspn.py --dataset forest10 --skew 0.0 --corr 0.0 --update-method adaincr --update-query-root update --update-skew 0.0 --update-corr 0.0

If you want to test update method *AdaIncr* on Query-Update-Only workload, the command is

    python scripts/run_qspn.py --dataset forest10 --skew 0.0 --corr 0.0 --update-method adaincr --update-query-root template --update-skew 0.5 --update-corr 0.5

You can also change the dataset by change parameter '--dataset forest10' to '--dataset gas/census13/power7' and so on.

The result will be outputed to file *'qspn_update_expr.log'*, including workload executing time and Mean Q-error on each interval of the whole update workload (we set $10$ checkpoints).

### M-QSPN Multi-Table CardEst

Dataset: IMDB, Workload: JOB-light

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --inference

The result will be outputed to the console, including Model Size, Mean Inference Time, Q-error of each query, 50th/90th/95th/99th/Max/Mean Q-error of CardEst on the workload.

If you want construction a *M-QSPN* model by yourself, such as a model with binning size as $1000$, the command is

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --model-binning-size 1000 --train

Also, if you want to test this model (with not default binning size: $1000$), the command is

    python scripts/run_mqspn.py --dataset job --workload-trainset mscn_queries_neurocard_format --workload-testset job-light --model-binning-size 1000 --inference

## Code Structure

```sh
|-- data # folder of datasets and workloads
|-- models # folder of constructed QSPN & M-QSPN models
|-- qspn # folder of QSPN & M-QSPN source code
|-- scripts # folder of scripts to run CardEst
    |-- run_qspn.py # Single-Table CardEst & Model Update by QSPN model
    |-- run_mqspn.py # Multi-Table CardEst by M-QSPN model
    |-- update_eval.py # (Please Do NOT Run) Source code for model update method, called by run_qspn.py
|-- settings.py # (Please Do NOT Modify) Configuration of Paths
```

## Advanced Setting

We first introduce the file formats of data files and query workload files which you can refer to the files in *'data/'* and then illustrate the running parameters.

### Single-Table Dataset

For a new single-table dataset, supposing whose name is 'sgtbdat', we firstly create a directory with the same name in the path *'data/'*. So we get a path *'data/sgtbdat'* now. Then we prepare a csv, an hdf and a json file in the following.

**File 'data/sgtbdat/data.csv':**

The first line contains names of all columns with delimiter ','. In the following lines, each line is a data tuple (record) which contains values of all columns with delimiter ','. Note that the encoding of this file should be 'utf-8' and its endline should be '\\n'.

**File 'data/sgtbdat/data.hdf':**

The content of this file is the same with *'data/sgtbdat/data.csv'* but its type is pandas.DataFrame. We create pandas.DataFrame 'df' whose columns are the same with the firstline of *'data/sgtbdat/data.csv'* and each row is a data tuple. Afterwards, we create and read this file by:

    df.to_hdf('data/sgtbdat/data.hdf', key='dataframe')
    df = pandas.read_hdf('data/sgtbdat/data.hdf', key='dataframe')

**File 'data/sgtbdat/meta.json':**

This json file has two keys "columns" and "cardinality". The value of key "columns" is a dict where the name of each column maps to its index starting from $0$ in the order of columns. And the value of key "cardinality" is the total count of the data tuples.

Finally, we create an empty directory *'data/sgtbdat/queries'* for query workloads.

### Single-Table Workload

For a new single-table workload for dataset 'sgtbdat', supposing whose name is 'user', we firstly create a directory with the same name in the path *'data/sgtbdat/queries'*. So we get a path *'data/sgtbdat/queries/user'*. The we prepare some npy files.

**File 'data/sgtbdat/queries/user/test_query_sc.npy':**

The content of this file is a 3-dimensonal numpy array 'arr_query'. The first value of the shape is equal to the number of queries in this workload, representing each query. The second value of the shape is equal the number of data columns, represent query predicate on each column. The third value of the shape is $2$, representing the lower bound constraint and upper bound constraint of this query predicate on this column. Afterwards, we create and read this file by:

    numpy.save('data/sgtbdat/queries/user/test_query_sc.npy', arr_query)
    numpy.load('data/sgtbdat/queries/user/test_query_sc.npy', arr_query)

If this query workload is only used as update workload, only this one file is enough. But please note that if the $q$-th (staring from $0$) query is a data update operation (INSERT sentence), the 'arr_query[q, :, 1]' must be all $-1$ and the 'arr_query[q, :, 0]' represents the inserted data tuple. This is how we distinguish between data update operation and query.

For query workload used for Single-Table CardEst, it should contains another three npy files. 

**File 'data/sgtbdat/queries/user/test_true_sc.npy':**

The content of this file is a 1-dimensonal numpy array 'arr_truecard'. The first value of the shape is equal to the number of queries in this workload, representing the true cardinality of each query. Afterwards, we create and read this file by:

    numpy.save('data/sgtbdat/queries/user/test_true_sc.npy', arr_truecard)
    numpy.load('data/sgtbdat/queries/user/test_true_sc.npy', arr_truecard)

Another two numpy files *'data/sgtbdat/queries/user/train_query_sc.npy'* and *'data/sgtbdat/queries/user/train_true_sc.npy'* shares the same format with *'data/sgtbdat/queries/user/test_query_sc.npy'* and *'data/sgtbdat/queries/user/test_true_sc.npy' respectively.

### Multi-Table Dataset

For a new multi-table dataset, supposing whose name is 'multbdat', we firstly create a directory to get the path *'data/multbdat'*. Then we prepare multiple csv files for all tables of this dataset in the same way of single-table dataset. And we also create an empty directory *'data/multbdat/queries'* for query workloads.

### Multi-Table Workload

For a multi-table workload for dataset 'multbdat', suppoing whose name is 'userb', we prepare the csv file *'data/multbdat/queries/userb.csv'* in which each line represents a multi-table query. And a multi-table query contains four parts with delimiter '#'.

The first part contains the names of all tables accessed by the query with delimiter ','. The second part contains all join predicates of the query with delimiter ',' where each join predicate is described as 'tableA.columnX=tableB.columnY'. The third part contains all range predicates of the query with delimiter ',' between each symbol such as 'tableA.columnI,=,100,tableB.columnJ,>,20,tableB.columnJ,<,50'. The four part is a number, the true cardinality of the query.

The workload train set and test set is not distinguished by specific file name. For example, you can construct a *M-QSPN* model on workload 'userb' and test it on workload 'userc'.

### Running Parameter

#### QSPN Paramter

**--dataset**: set the name of single-table dataset like '--dataset sgtbdat'

**--query-path**: set the name of single-table workload like '--query-path user'

**--Nx**, **--lower**, **--upper**: tune the parameters of adaptive RDC threshold of *QSPN* model and we recommends to keep the default setting '--Nx 5.0 --lower 0.1 --upper 0.3' which we think is OK for various datasets.

If you want to construct a *QSPN* model, add a switch parameter **--train**, the model file will be dumped to the file like *'models/single_tables/qspn/qspn_5.0_0.1_0.3_QSplit_sgtbdat_user.pkl'*. Correspondingly, if you want to test this constructed model, add a switch parameter **--inference**. And if you want to test the *QSPN* Update Method by constructing model on workload 'user1' and testing model on workload 'user2', you should also set parameter **--update-query-path** like '--query-path user1 --update-query-path user2' and **--update-method** like '--update-method adaincr' with a switch parameter **--update**.

#### M-QSPN Parater

**--dataset**: set the name of multi-table dataset like '--dataset multbdat'

**--workload-trainset**: set the name of multi-table workload train set like '--workload-trainset userb'

**--workload-testset**: set the name of multi-table workload test set like '--workload-testset userc'

**--model-binning-size**: tune the bins size limit of *M-QSPN* model binning like '--model-binning-size 200'. A bigger setting usually improves the CardEst accuracy of *M-QSPN* model with longer inference time.

If you want to construct a *M-QSPN* model, add a switch parameter **--train**, the model file will be dumped to the file like *'models/multi_tables/mqspn/mqspn_multbdat_userb_200.pkl'*. Correspondingly, if you want to test this constructed model, add a switch parameter **--inference**. Note that our code actually shows that we do not use workload test set for model construction and we only use the name of workload train set to locate the path of model file for model test though 'scripts/run_mqspn.py' always needs setting both '--workload-trainset' and '--workload-testset'.

<!-- ## Acknowledgement

... -->