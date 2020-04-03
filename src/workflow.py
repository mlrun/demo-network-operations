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
        inplace = False,
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
        score_method = 'micro',
        config_filepath = 'store://network-operations/lgb_configs',
    ):
    
    describe = funcs['describe'].as_step(name='describe-raw-data',
                                                handler="summarize",  
                                                params={"key": "summary", 
                                                        "label_column": label_column, 
                                                        'class_labels': ['0', '1'],
                                                        'plot_hist': True,
                                                        'plot_dest': 'plots'},
                                                inputs={"table": df_artifact},
                                                outputs=["summary", "scale_pos_weight"])
    
    # Run preprocessing on the data
    aggregate = funcs['aggregate'].as_step(name='aggregate',
                                                  params={'metrics': metrics,
                                                          'labels': labels,
                                                          'metric_aggs': metric_aggs,
                                                          'label_aggs': label_aggs,
                                                          'suffix': suffix,
                                                          'inplace': inplace,
                                                          'window': window,
                                                          'center': center,
                                                          'save_to': save_to},
                                                  inputs={'df_artifact': df_artifact},
                                                  outputs=['aggregate'],
                                                  handler='aggregate')

    describe = funcs['describe'].as_step(name='describe-feature-vector',
                                                handler="summarize",  
                                                params={"key": "summary", 
                                                        "label_column": label_column, 
                                                        'class_labels': ['0', '1'],
                                                        'plot_hist': True,
                                                        'plot_dest': 'plots'},
                                                inputs={"table": aggregate.outputs['aggregate']},
                                                outputs=["summary", "scale_pos_weight"])
    
    train = funcs['train'].as_step(name='train', 
                                          handler='train_model',
                                          params={'sample'          : SAMPLE_SIZE,
                                                  'label_column'    : label_column,
                                                  'test_size'       : TEST_SIZE,
                                                  'train_val_split' : TRAIN_VAL_SPLIT,
                                                  'rng'             : RNG,
                                                  'score_method'    : score_method},
                                          inputs={"dataset": aggregate.outputs['aggregate'],
                                                  'model_pkg_file' : config_filepath},
                                          outputs=['model', 'test_set'])
    
    test = funcs['test'].as_step(name='test',
                                 handler='test_classifier',
                                 params={'label_column': label_column},
                                 inputs={'models_path': train.outputs['model'],
                                         'test_set': train.outputs['test_set']})
    
    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(project='nuclio-serving',
                                                 models={'predictor': train.outputs['model']})
