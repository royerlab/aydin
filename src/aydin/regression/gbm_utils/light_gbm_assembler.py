"""Custom LightGBM model assembler for m2cgen code generation.

This module provides :class:`LightGBMModelAssembler`, which extends the
m2cgen tree-boosting assembler to support optional tree slicing for
partial model export.
"""

from m2cgen import ast
from m2cgen.assemblers import utils
from m2cgen.assemblers.boosting import (
    LEAVES_CUTOFF_THRESHOLD,
    BaseTreeBoostingAssembler,
)


class LightGBMModelAssembler(BaseTreeBoostingAssembler):
    """Assembler that converts a LightGBM model into an m2cgen AST.

    Extends the base tree-boosting assembler with support for an optional
    ``tree_slice`` parameter that allows exporting only a subset of the
    decision trees.

    Attributes
    ----------
    classifier_names : set
        Set of classifier class names recognised by this assembler.
    n_iter : int
        Number of boosting iterations represented.
    average_output : bool
        Whether the model averages the output of all trees.
    """

    classifier_names = {"LGBMClassifier"}

    def __init__(
        self, model, leaves_cutoff_threshold=LEAVES_CUTOFF_THRESHOLD, tree_slice=None
    ):
        """Initialise the assembler from a LightGBM model.

        Parameters
        ----------
        model : lightgbm.Booster
            Trained LightGBM model.
        leaves_cutoff_threshold : int
            Maximum number of leaves before switching to a different
            assembly strategy.
        tree_slice : slice, optional
            If provided, only the trees in this slice range are assembled.
        """
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
        """Apply final averaging transformation if the model uses average output.

        Parameters
        ----------
        ast_to_transform : m2cgen.ast.Expr
            The AST expression to transform.

        Returns
        -------
        m2cgen.ast.Expr
            Transformed AST expression.
        """
        if self.average_output:
            coef = 1 / self.n_iter
            return utils.apply_bin_op(
                ast_to_transform, ast.NumVal(coef), ast.BinNumOpType.MUL
            )
        else:
            return super()._final_transform(ast_to_transform)

    def _assemble_tree(self, tree):
        """Recursively assemble a single decision tree into an AST.

        Parameters
        ----------
        tree : dict
            Tree structure dictionary from LightGBM's ``dump_model()`` output.

        Returns
        -------
        m2cgen.ast.Expr
            AST representation of the decision tree.
        """
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
