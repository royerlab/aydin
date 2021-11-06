from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.boosting import (
    BaseTreeBoostingAssembler,
    LEAVES_CUTOFF_THRESHOLD,
)


class LightGBMModelAssembler(BaseTreeBoostingAssembler):

    classifier_names = {"LGBMClassifier"}

    def __init__(
        self, model, leaves_cutoff_threshold=LEAVES_CUTOFF_THRESHOLD, tree_slice=None
    ):
        model_dump = model.dump_model()
        if tree_slice is None:
            trees = [m["tree_structure"] for m in model_dump["tree_info"]]
        else:
            trees = [m["tree_structure"] for m in model_dump["tree_info"][tree_slice]]
            # [model_dump["tree_info"][tree_slice]["tree_structure"]]

        self.n_iter = len(trees) // model_dump["num_tree_per_iteration"]
        self.average_output = model_dump.get("average_output", False)

        super().__init__(model, trees, leaves_cutoff_threshold=leaves_cutoff_threshold)

    def _final_transform(self, ast_to_transform):
        if self.average_output:
            coef = 1 / self.n_iter
            return utils.apply_bin_op(
                ast_to_transform, ast.NumVal(coef), ast.BinNumOpType.MUL
            )
        else:
            return super()._final_transform(ast_to_transform)

    def _assemble_tree(self, tree):
        if "leaf_value" in tree:
            return ast.NumVal(tree["leaf_value"])

        threshold = ast.NumVal(tree["threshold"])
        feature_ref = ast.FeatureRef(tree["split_feature"])

        op = ast.CompOpType.from_str_op(tree["decision_type"])
        assert op == ast.CompOpType.LTE, "Unexpected comparison op"

        # Make sure that if the "default_left" is true the left tree branch
        # ends up in the "else" branch of the ast.IfExpr.
        if tree["default_left"]:
            op = ast.CompOpType.GT
            true_child = tree["right_child"]
            false_child = tree["left_child"]
        else:
            true_child = tree["left_child"]
            false_child = tree["right_child"]

        return ast.IfExpr(
            ast.CompExpr(feature_ref, threshold, op),
            self._assemble_tree(true_child),
            self._assemble_tree(false_child),
        )
