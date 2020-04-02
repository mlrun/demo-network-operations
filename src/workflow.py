from kfp import dsl
from mlrun import mount_v3io
import os

funcs = {}

def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        # Add V3IO Mount
        f.apply(mount_v3io())
        
        # Always pull images to keep updates
        f.spec.image_pull_policy = 'Always'
        
    functions['aggregate'].spec.image = 'mlrun/ml-models:0.4.6'
    for fn, fv in functions.items():
        print(f'Function: {fn}')
        print(fv.spec)
@dsl.pipeline(
    name='Network Operations Demo',
    description='Train a Failure Prediction LGBM Model over sensor data'
)
def kfpipeline(
        df_artifact = 'store://network-operations/metrics',
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
        config_filepath = 'store://network-operations/lgb_configs',
    ):
    
    describe = funcs['describe'].as_step(name='describe-raw-data',
                                                handler="summarize",  
                                                params={"key": "summary", 
                                                        "label_column": "is_error", 
                                                        'class_labels': ['0', '1'],
                                                        'plot_hist': True,
                                                        'plot_dest': 'plots'},
                                                inputs={"table": df_artifact},
                                                outputs=["summary", "scale_pos_weight"])
    
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
                                                  image='mlrun/mlrun')

    describe = funcs['describe'].as_step(name='describe-feature-vector',
                                                handler="summarize",  
                                                params={"key": "summary", 
                                                        "label_column": "is_error", 
                                                        'class_labels': ['0', '1'],
                                                        'plot_hist': True,
                                                        'plot_dest': 'plots'},
                                                inputs={"table": aggregate.outputs['aggregate']},
                                                outputs=["summary", "scale_pos_weight"])
    
    train = funcs['train'].as_step(name='train', 
                                          handler='train_model',
                                          params={'model_pkg_class' : config_filepath,
                                                  'sample'          : -1,
                                                  'label_column'    : "is_error",
                                                  'test_size'       : 0.10,
                                                  'train_val_split' : 0.75,
                                                  'rng'             : 1},
                                          inputs={"data_key": aggregate.outputs['aggregate']},
                                          outputs=['model', 'test-set'])
    
#     test = funcs['test'].as_step()
    
    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(project='nuclio-serving',
                                                 models={'predictor': train.outputs['model']})
