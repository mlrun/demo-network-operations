from kfp import dsl
from mlrun import mount_v3io, mlconf
import os
from nuclio.triggers import V3IOStreamTrigger

funcs = {}
projdir = os.getcwd()
label_column = 'is_error'
model_inference_stream = '/users/admin/demo-network-operations/streaming/inference_stream'
model_inference_url = f'http://v3io-webapi:8081{model_inference_stream}'


def init_functions(functions: dict, project=None, secrets=None):
    tags = {'ml-models': 'unstable-py36',
            'mlrun': 'unstable'}
    for f in functions.values():
        # Add V3IO Mount
        f.apply(mount_v3io())
        if hasattr(f.spec, 'image'):
            if not f.spec.image.endswith('unstable-py36') and (not f.spec.image == ''):
                current_image = f.spec.image.split(':')[0]
                fn_image = f"{current_image}:{tags[current_image.split('/')[1]]}"
                print(fn_image)
                f.spec.image = fn_image
        
        # Always pull images to keep updates
        f.spec.image_pull_policy = 'Always'
        
    functions['serving'].metadata.name = 'netops-server'
    functions['serving'].spec.min_replicas = 1
    functions['serving'].spec.build.baseImage = 'mlrun/mlrun:unstable'
    
    # Define inference-stream related triggers
    functions['serving'].set_envs({'INFERENCE_STREAM': model_inference_stream})
    functions['s2p'].add_trigger('labeled_stream', V3IOStreamTrigger(url=f'{model_inference_url}@s2p'))
    functions['s2p'].set_envs({'window': 10,
                               'features': 'cpu_utilization',
                               'save_to': '/User/demo-network-operations/streaming/inference_pq/',
                               'base_dataset': '/User/demo-network-operations/artifacts/test_set_preds.parquet',
                               'hub_url': '/User/functions/{name}/function.yaml',
                               'mount_path': '~/',
                               'mount_remote': '/User'})
                
        
@dsl.pipeline(
    name='Network Operations Demo',
    description='Train a Failure Prediction LGBM Model over sensor data'
)
def kfpipeline(
        df_artifact = '/User/demo-network-operations/data/metrics.pq',
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
        aggregate_fn_url = '/User/functions/aggregate/function.yaml',
        streaming_metrics_table = '/User/demo-network-operations/streaming/metrics',
        streaming_features_table = '/User/demo-network-operations/streaming/features',
        streaming_predictions_table = '/User/demo-network-operations/streaming/predictions',
        streaming_labeled_table = '/users/admin/demo-network-operations/streaming/labeled_stream',
        predictions_col = 'predictions',
        secs_to_generate = 10,
        deploy_streaming = True,
        deploy_concept_drift = True
    ):
    
    # Run preprocessing on the data
    aggregate = funcs['aggregate'].as_step(name='aggregate',
                                                  params={'metrics': metrics,
                                                          'metric_aggs': metric_aggs,
                                                          'suffix': suffix,
                                                          },
                                                  inputs={'df_artifact': df_artifact},
                                                  outputs=['aggregate'],
                                                  handler='aggregate',
                                                  image='mlrun/ml-models:unstable-py36')

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
                                                           image='mlrun/ml-models:unstable-py36')
    
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
                                   image='mlrun/ml-models:unstable-py36')
    
    test = funcs['test'].as_step(name='test',
                                 handler='test_classifier',
                                 params={'label_column': label_column,
                                         'predictions_column': predictions_col},
                                 inputs={'models_path': train.outputs['model'],
                                         'test_set': train.outputs['test_set']},
                                 image='mlrun/ml-models:unstable-py36')
    
    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(env={'model_path': train.outputs['model'],
                                               'FEATURES_TABLE': streaming_features_table,
                                               'PREDICTIONS_TABLE': streaming_predictions_table,
                                               'prediction_col': predictions_col}, 
                                          tag='v1')
    
    # test out new model server (via REST API calls)
    tester = funcs["model_server-tester"].as_step(name='model-tester',
                                                  params={'addr': deploy.outputs['endpoint'], 
                                                          'model': "predictor",
                                                          'label_column': label_column},
                                                  inputs={'table': train.outputs['test_set']})
    
    with dsl.Condition(deploy_streaming == True):
    
        # Streaming demo functions
        preprocessor = funcs['preprocessor'].deploy_step(env={ 'aggregate_fn_url': aggregate_fn_url,
                                                                'METRICS_TABLE': streaming_metrics_table,
                                                                'FEATURES_TABLE': streaming_features_table,
                                                                'metrics': metrics,
                                                                'metric_aggs': metric_aggs,
                                                                'suffix': suffix,
                                                                'base_dataset': '/User/demo-network-operations/artifacts/selected_features.parquet',
                                                                'label_col': label_column}).after(tester)

        labeled_stream_creator = funcs['labeled_stream'].deploy_step(env={'METRICS_TABLE': streaming_metrics_table,
                                                                                  'PREDICTIONS_TABLE': streaming_predictions_table,
                                                                                  'OUTPUT_STREAM': streaming_labeled_table,
                                                                                  'label_col': label_column,
                                                                                  'prediction_col': predictions_col}).after(tester)

        generator = funcs['generator'].deploy_step(env={'SAVE_TO': streaming_metrics_table,
                                                        'SECS_TO_GENERATE': secs_to_generate}).after(preprocessor)
        
        with dsl.Condition(deploy_concept_drift == True):

            concept_builder = funcs['concept_drift'].deploy_step(skip_deployed=True)

            concept_drift = funcs['concept_drift'].as_step(name='concept_drift_deployer',
                                                           params={'models': ['ddm', 'eddm', 'pagehinkley'],
                                                                   'label_col': label_column,
                                                                   'prediction_col': predictions_col,
                                                                   'hub_url': '/User/functions/{name}/function.yaml',
                                                                   'output_tsdb': '/User/network-operations/streaming/drift_tsdb',
                                                                   'input_stream': 'http://v3io-webapi:8081/users/admin/network-operations/streaming/labeled_stream@cd2',
                                                                   'output_stream': '/User/network-operations/streaming/drift_stream'},
                                                           inputs={'base_dataset': '/User/demo-network-operations/artifacts/test_set_preds.parquet'},
                                                           artifact_path=mlconf.artifact_path,
                                                           image=concept_builder.outputs['image']).after(labeled_stream_creator)

            s2p = funcs['s2p'].deploy_step(env={'window': 10,
                                                'features': 'cpu_utilization',
                                                'save_to': '/User/demo-network-operations/streaming/inference_pq/',
                                                'base_dataset': '/User/demo-network-operations/artifacts/test_set_preds.parquet',
                                                'hub_url': '/User/functions/{name}/function.yaml',
                                                'mount_path': '~/',
                                                'mount_remote': '/User'}).after(tester)
    
