from kfp import dsl
import kfp
from mlrun import mount_v3io
import urllib
import json
import os

funcs = {}
lgbm_url = urllib.request.urlopen('https://raw.githubusercontent.com/yjb-ds/lightgbm-project/pre-project/lightgbm-conf.json')
LGBM = json.load(lgbm_url)
project_dir = '/User/demo-network-operations'

def init_functions(functions: dict, params=None, secrets=None):
    '''
    This function will run before running the project.
    It allows us to add our specific system configurations to the functions
    like mounts or secrets if needed.

    In this case we will add Iguazio's user mount to our functions using the
    `mount_v3io()` function to automatically set the mount with the needed
    variables taken from the environment. 
    * mount_v3io can be replaced with mlrun.platforms.mount_pvc() for 
    non-iguazio mount

    @param functions: <function_name: function_yaml> dict of functions in the
                        workflow
    @param params: parameters for the function configurations
    @param secrets: secrets required for the functions for s3 connections and
                    such
    '''
    for f in functions.values():
        f.apply(mount_v3io())                  # On Iguazio (Auto-mount /User)
        # f.apply(mlrun.platforms.mount_pvc()) # Non-Iguazio mount


@kfp.dsl.pipeline(
    name='Network Operations Demo',
    description='Train a Failure Prediction LGBM Model over sensor data'
)
def kfpipeline(
        df_artifact = os.path.join(project_dir, 'data', 'metrics.pq'),
        metrics = ['cpu_utilization'],
        labels = ['is_error'],
        metric_aggs = ['mean', 'sum'],
        label_aggs = ['max'],
        suffix = 'daily',
        append_to_df = True,
        window = 5,
        center = True,
        save_to = os.path.join('data', 'aggregate.pq'),
        describe_table = 'summary',
        label_column = 'is_error',
        class_labels = [1, 0],
        SAMPLE_SIZE      = -1, # -n for random sample of n obs, -1 for entire dataset, +n for n consecutive rows
        TEST_SIZE        = 0.1,       # 10% set aside
        TRAIN_VAL_SPLIT  = 0.75,      # remainder split into train and val
        RNG              = 1,
        class_params = LGBM['CLASS_PARAMS'],
        fit_params = LGBM['FIT_PARAMS'],
    ):
    
    # Run preprocessing on the data
    aggregate = funcs['aggregate'].as_step(name='aggregate',
                                                  params={'df_artifact': df_artifact,
                                                          'metrics': metrics,
                                                          'labels': labels,
                                                          'metric_aggs': metric_aggs,
                                                          'label_aggs': label_aggs,
                                                          'suffix': suffix,
                                                          'append_to_df': append_to_df,
                                                          'window': window,
                                                          'center': center,
                                                          'save_to': save_to},
                                                  outputs=['aggregate'],
                                                  handler='aggregate',
                                                  image='docker-registry.default-tenant.app.cnyidfihnjsz.iguazio-cd0.com:80/mlrun/func-default-aggregate-latest')

    describe = funcs['describe'].as_step(name='describe',
                                                handler="table_summary",  
                                                params={"key": describe_table, 
                                                        "label_column": label_column, 
                                                        'class_labels': class_labels},
                                                inputs={"table": aggregate.outputs['aggregate']},
                                                outputs=["summary", "scale_pos_weight"]).after(aggregate)
    
    train = funcs['train_lgbm'].as_step(name='train',
                                               handler='train_model',
                                               params={'sample'          : SAMPLE_SIZE,
                                                       'label_column'    : label_column,
                                                       'test_size'       : TEST_SIZE,
                                                       'train_val_split' : TRAIN_VAL_SPLIT,
                                                       'rng'             : RNG,
                                                       'class_params'    : class_params,
                                                       'fit_params'      : fit_params},
                                               inputs={"data_key": aggregate.outputs['aggregate'],
                                                        "scale_pos_weigth": describe.outputs["scale_pos_weight"]},
                                               outputs=['model', 'test-set'])
    

    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(project='nuclio-serving',
                                                 models={'predictor': train.outputs['model']})