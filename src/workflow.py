from kfp import dsl
from mlrun import mount_v3io, mlconf
import os
from nuclio.triggers import V3IOStreamTrigger

funcs = {}
projdir = os.getcwd()
label_column = 'is_error'
model_inference_stream = '/bigdata/network-operations/inference_stream'
model_inference_url = f'http://v3io-webapi:8081{model_inference_stream}'


def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        # Add V3IO Mount
        f.apply(mount_v3io())
        
        # Always pull images to keep updates
        f.spec.image_pull_policy = 'Always'
        
    functions['serving'].metadata.name = 'netops-server'
    functions['serving'].spec.min_replicas = 1
    functions['serving'].spec.build.baseImage = 'mlrun/mlrun:0.4.7'
    
    # Define inference-stream related triggers
    functions['serving'].set_envs({'INFERENCE_STREAM': model_inference_stream})
    functions['s2p'].add_trigger('labeled_stream', V3IOStreamTrigger(url=f'{model_inference_url}@s2p'))
    functions['s2p'].set_envs({'window': 10,
                               'features': 'cpu_utilization',
                               'save_to': '/bigdata/inference_pq/',
                               'base_dataset': 'store://network-operations/test_test_set_preds',
                               'hub_url': '/User/functions/{name}/function.yaml',
                               'mount_path': '/bigdata',
                               'mount_remote': '/bigdata'})
                
        
@dsl.pipeline(
    name='Network Operations Demo',
    description='Train a Failure Prediction LGBM Model over sensor data'
)
def kfpipeline(
        df_artifact = 'store://network-operations/metrics',
        metrics = ['cpu_utilization', 'throughput', 'packet_loss', 'latency'],
        metric_aggs = ['mean', 'sum', 'std', 'var', 'min', 'max', 'median'],
        suffix = 'daily',
        window = 20,
        describe_table = 'netops',
        describe_sample = 0.3,
        label_column = label_column,
        class_labels = [1, 0],
        SAMPLE_SIZE      = -1, # -n for random sample of n obs, -1 for entire dataset, +n for n consecutive rows
        TEST_SIZE        = 0.1,       # 10% set aside
        TRAIN_VAL_SPLIT  = 0.75,      # remainder split into train and val
    ):
    
    describe = funcs['describe'].as_step(name='describe-raw-data',
                                                handler="summarize",  
                                                params={"key": f"{describe_table}_raw", 
                                                        "label_column": label_column, 
                                                        'class_labels': ['0', '1'],
                                                        'plot_hist': True,
                                                        'plot_dest': 'plots/raw',
                                                        'sample': describe_sample},
                                                inputs={"table": df_artifact},
                                                outputs=["summary", "scale_pos_weight"])
    
    # Run preprocessing on the data
    aggregate = funcs['aggregate'].as_step(name='aggregate',
                                                  params={'metrics': metrics,
                                                          'metric_aggs': metric_aggs,
                                                          'suffix': suffix,
                                                          },
                                                  inputs={'df_artifact': df_artifact},
                                                  outputs=['aggregate'],
                                                  handler='aggregate',
                                                  image='mlrun/ml-models:0.4.7')

    describe = funcs['describe'].as_step(name='describe-aggregation',
                                                handler="summarize",  
                                                params={"key": f"{describe_table}_aggregate", 
                                                        "label_column": label_column, 
                                                        'class_labels': class_labels,
                                                        'plot_hist': True,
                                                        'plot_dest': 'plots/aggregation',
                                                        'sample': describe_sample},
                                                inputs={"table": aggregate.outputs['aggregate']},
                                                outputs=["summary", "scale_pos_weight"])
    
    feature_selection = funcs['feature_selection'].as_step(name='feature_selection',
                                                           handler='feature_selection',
                                                           params={'k': 5,
                                                                   'min_votes': 3,
                                                                   'label_column': label_column},
                                                           inputs={'df_artifact': aggregate.outputs['aggregate']},
                                                           outputs=['feature_scores', 
                                                                    'max_scaled_scores_feature_scores'
                                                                    'selected_features_count', 
                                                                    'selected_features'],
                                                           image='mlrun/ml-models:0.4.7')
    
    describe = funcs['describe'].as_step(name='describe-feature-vector',
                                            handler="summarize",  
                                            params={"key": f'{describe_table}_features', 
                                                    "label_column": label_column, 
                                                    'class_labels': class_labels,
                                                    'plot_hist': True,
                                                    'plot_dest': 'plots/feature_vector'},
                                            inputs={"table": feature_selection.outputs['selected_features']},
                                            outputs=["summary", "scale_pos_weight"])
    
    train = funcs['train'].as_step(name='train',
                                   params={"sample"          : SAMPLE_SIZE, 
                                           "label_column"    : label_column,
                                           "test_size"       : TEST_SIZE},
                                   inputs={"dataset"         : feature_selection.outputs['selected_features']},
                                   hyperparams={'model_pkg_class': ["sklearn.ensemble.RandomForestClassifier", 
                                                                    "sklearn.linear_model.LogisticRegression",
                                                                    "sklearn.ensemble.AdaBoostClassifier"]},
                                   selector='max.accuracy',
                                   outputs=['model', 'test_set'],
                                   image='mlrun/ml-models:0.4.7')
    
    test = funcs['test'].as_step(name='test',
                                 handler='test_classifier',
                                 params={'label_column': label_column},
                                 inputs={'models_path': train.outputs['model'],
                                         'test_set': train.outputs['test_set']},
                                 image='mlrun/ml-models:0.4.7')
    
    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(models={'predictor': train.outputs['model']}, tag='v1')
    
    # test out new model server (via REST API calls)
    tester = funcs["model_server-tester"].as_step(name='model-tester',
                                                  params={'addr': deploy.outputs['endpoint'], 
                                                          'model': "predictor",
                                                          'label_column': label_column},
                                                  inputs={'table': train.outputs['test_set']})
    
    concept_drift = funcs['concept_drift'].as_step(name='concept_drift_deployer',
                                                   params={'models': ['ddm', 'eddm', 'pagehinkley'],
                                                           'label_col': 'is_error',
                                                           'prediction_col': 'prediction',
                                                           'hub_url': '/User/functions/{name}/function.yaml',
                                                           'output_tsdb': '/bigdata/network-operations/drift_tsdb',
                                                           'input_stream': 'http://v3io-webapi:8081/bigdata/network-operations/inference_stream@cd2',
                                                           'output_stream': '/bigdata/network-operations/drift_stream'},
                                                   inputs={'base_dataset': 'store://network-operations/test_test_set_preds'},
                                                   artifact_path=mlconf.artifact_path,
                                                   image='mlrun/ml-models:0.4.7').after(deploy)
    
    s2p = funcs['s2p'].deploy_step(project='network-operations').after(deploy)
