from flask_restx import reqparse, fields

parserWandb = reqparse.RequestParser(bundle_errors=True)
parserWandb.add_argument("key",
                         required=True,
                         default=None,
                         help="authentication key for W&B")

parserAdd = reqparse.RequestParser(bundle_errors=True)
parserAdd.add_argument("name",
                       required=True,
                       default='my_model',
                       help="Used as a key in local models storage; Must be unique;")

parserAdd.add_argument("backbone_type",
                       required=True,
                       default="resnet18",
                       help="resnet18/resnet50")

parserAdd.add_argument("device",
                       required=True,
                       help="cpu/cuda",
                       default="cpu")


parserRemove = reqparse.RequestParser(bundle_errors=True)
parserRemove.add_argument("name",
                          type=str,
                          required=True,
                          help="Name of a model you want to remove",
                          location="args")

parserTrain = reqparse.RequestParser(bundle_errors=True)
parserTrain.add_argument("project_name",
                         type=str,
                         required=False,
                         default='rest_api',
                         help="Name of W&B project",
                         location="args")

parserTrain.add_argument("experiment_name",
                         type=str,
                         required=False,
                         help="Name of W&B experiment",
                         location="args")

parserTrain.add_argument("model_name",
                         type=str,
                         required=True,
                         default='my_model',
                         help="Name of a model you want to train",
                         location="args")

parserTrain.add_argument("dataset_path",
                         type=str,
                         required=True,
                         help="Path to dataset",
                         location="args")

parserTrain.add_argument("valid_part",
                         type=float,
                         required=False,
                         help="What fraction of data goes to validation part",
                         default=0.1,
                         location="args")

parserTrain.add_argument("epochs_numb",
                         type=int,
                         required=True,
                         help="Number of epochs",
                         default=20,
                         location="args")

parserTrain.add_argument("batch_size",
                         type=int,
                         required=False,
                         default=32,
                         help="Batch size",
                         location="args")

parserTrain.add_argument("optimizer_name",
                         type=str,
                         required=False,
                         default='Adam',
                         help="Optimizer name",
                         location="args")

parserTrain.add_argument("learning_rate",
                         type=float,
                         required=False,
                         default=0.01,
                         help="Initial learning rate",
                         location="args")

parserTrain.add_argument("freeze_backbone",
                         type=bool,
                         required=True,
                         help="Freeze backbone",
                         location="args")

parserTest = reqparse.RequestParser(bundle_errors=True)
parserTest.add_argument("name",
                        type=str,
                        required=True,
                        help="Name of a model you want to test",
                        location="args")

parserTest.add_argument("dataset_path",
                        type=str,
                        required=True,
                        help="Path to dataset",
                        location="args")


parserPredict = reqparse.RequestParser(bundle_errors=True)
parserPredict.add_argument("name",
                        type=str,
                        default='my_model',
                        required=True,
                        help="Name of a model you want to use for prediction",
                        location="args")

parserPredict.add_argument("dataset_path",
                        type=str,
                        required=True,
                        help="Path to data (directory with images)",
                        location="args")
