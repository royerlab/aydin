import warnings
from operator import gt, lt

from lightgbm.callback import EarlyStopException, _format_eval_result

from aydin.util.log.log import lprint


def early_stopping(gbm_regressor, stopping_rounds):
    """Create a callback that activates early stopping.

    Note
    ----
    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.

    Parameters
    ----------

    gbm_regressor : GBMRegressor
        parent regressor
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.


    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
    cmp_op = []
    enabled = [True]

    def _init(env):
        enabled[0] = not any(
            (boost_alias in env.params and env.params[boost_alias] == 'dart')
            for boost_alias in ('boosting', 'boosting_type', 'boost')
        )
        if not enabled[0]:
            warnings.warn('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError(
                'For early stopping, '
                'at least one dataset and eval metric is required for evaluation'
            )

        msg = "Training until validation scores don't improve for {} rounds."
        lprint(msg.format(stopping_rounds))

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            elif env.iteration - best_iter[i] >= stopping_rounds:
                lprint(
                    'Early stopping, best iteration is: [%d]\t%s'
                    % (
                        best_iter[i] + 1,
                        '\t'.join([_format_eval_result(x) for x in best_score_list[i]]),
                    )
                )
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                lprint(
                    'Did not meet early stopping. Best iteration is:\n[%d]\t%s'
                    % (
                        best_iter[i] + 1,
                        '\t'.join([_format_eval_result(x) for x in best_score_list[i]]),
                    )
                )
                raise EarlyStopException(best_iter[i], best_score_list[i])

            # This is where we stop training:
            if gbm_regressor._stop_fit:
                raise EarlyStopException(best_iter[i], best_score_list[i])

    _callback.order = 30
    return _callback
