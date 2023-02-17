import wandb
from flask import Flask
from flask_restx import Resource, Api

from api_parsers import *
from model_wrapper import ModelWrapper
from trainer import Trainer
from dataloader import load_data

MAX_MODEL_NUM = 10
MODELS_DICT = {}

app = Flask(__name__)
app.config["BUNDLE_ERRORS"] = True
api = Api(app)

model_predict = api.model(
    "Model.predict.input", {
        "img_url":
            fields.String(required=True,
                          title="Image URL",
                          description="")
    })

@api.route("/wandb/auth")
class WandAuth(Resource):
    @api.expect(parserWandb)
    @api.doc(responses={
        201: "Login succesful",
        404: "Login failed"
    })
    def post(self):
        success = wandb.login(key=parserWandb.parse_args()['key'])
        if success:
            return {"status": "login succesful"}, 201
        else:
            return {"status": "login failed"}, 404


@api.route("/models/list")
class ModelList(Resource):
    @api.doc(responses={201: "Success"})
    def get(self):
        return {
            "models": {
                i: {
                    "name": i,
                    "info":
                    MODELS_DICT[i].get_info(),
                }
                for i in MODELS_DICT.keys()
            }
        }, 201


@api.route("/models/add")
class ModelAdd(Resource):
    @api.expect(parserAdd)
    @api.doc(
        responses={
            201: "Success",
            401: "'params' error; Params must be a valid json or dict",
            402:
            "Error while initializing model; See description for more info",
            403: "Model with a given name already exists",
            408: "The max number of models has been reached"
        })
    def post(self):
        __name = parserAdd.parse_args()["name"]
        __type = parserAdd.parse_args()["backbone_type"]
        __device = parserAdd.parse_args()["device"]

        if len(MODELS_DICT) >= MAX_MODEL_NUM:
            return {
                "status":
                "Failed",
                "message":
                "The max number of models has been reached; You must delete one before creating another"
            }, 408

        if __name not in MODELS_DICT.keys():
            try:
                MODELS_DICT[__name] = ModelWrapper(__type, __device)
                return {"status": "OK", "message": f"Model {__name} created on {__device}!"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 402
        else:
            return {
                "status": "Failed",
                "message": "Model with a given name already exists"
            }, 403


@api.route("/models/remove")
class ModelRemove(Resource):
    @api.expect(parserRemove)
    @api.doc(responses={
        201: "Success",
        404: "Model with a given name does not exist"
    })
    def delete(self):
        __name = parserRemove.parse_args()["name"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist"
            }, 404
        else:
            MODELS_DICT.pop(__name)
            return {"status": "OK", "message": "Model removed!"}, 201


@api.route("/models/train")
class ModelTrain(Resource):
    @api.expect(parserTrain)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while training model; See description for more info"
        })

    def get(self):
        args = parserTrain.parse_args()

        if args['model_name'] not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                id2label, dataloaders = load_data(args["__dataset_path"], 32, args["valid_part"])
                config = {'optimizer_name': args["optimizer_name"],
                          "lr": args["learning_rate"],
                          'freeze_backbone': args["freeze_backone"]
                          }
                trainer = Trainer(config, MODELS_DICT[args['model_name']], dataloaders, id2label)
                if wandb.login():
                    wandb.init(
                        project=args["project_name"],
                        name=args()["experiment_name"],
                        config={
                            "dataset": args["__dataset_path"],
                            "optimizer": args["optimizer_name"],
                            "freeze_backbone": args["freeze_backone"],
                            "backbone_name": MODELS_DICT[args['model_name']].backbone_name,
                            "learnable_params": MODELS_DICT[args['model_name']].learnable_parameters,
                        })
                    wandb_url = wandb.run.get_url()
                else:
                    wandb_url = None
                MODELS_DICT[args['model_name']].wandb = {"project": args["project_name"],
                                                       "name": args["experiment_name"],
                                                       "url": wandb_url}
                trainer.train()
                return {"status": "OK", "message": f"Best train score {trainer.last_record}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406


@api.route("/models/test")
class ModelTest(Resource):
    @api.expect(parserTest)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while testing model; See description for more info"
        })

    def get(self):
        __name = parserTrain.parse_args()["name"]
        args = parserTrain.parse_args()
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                MODELS_DICT[__name].test(args["__dataset_path"])
                return {"status": "OK", "message": f"Test score {MODELS_DICT[__name].test_score}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406


@api.route("/models/predict")
class ModelPredict(Resource):
    @api.expect(model_predict)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            407: "Error while predicting result; See description for more info"
        })
    def post(self):
        __name = api.payload["name"]
        __params = api.payload
        __params.pop("name")
        print("BLAAAH")
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                pred = MODELS_DICT[__name].predict(__params)
                return {"result": f"predicted value: {pred}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 407


if __name__ == "__main__":
    app.run(debug=True)