import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tf_extended import math as tfe_math
import util

def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=tf.float32):
    """Creates a new local variable.
    Args:
        name: The name of the new or existing variable.
        shape: Shape of the new or existing variable.
        collections: A list of collection names to which the Variable will be added.
        validate_shape: Whether to validate the shape of the variable.
        dtype: Data type of the variables.
    Returns:
        The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [ops.GraphKeys.LOCAL_VARIABLES]
    return variables.Variable(
            initial_value=array_ops.zeros(shape, dtype=dtype),
            name=name,
            trainable=False,
            collections=collections,
            validate_shape=validate_shape)

def streaming_tp_fp_arrays(num_gbboxes, tp, fp, 
                           metrics_collections=None,
                           updates_collections=None,
                           name=None):
    """Streaming computation of True and False Positive arrays. 
    """
    with variable_scope.variable_scope(name, 'streaming_tp_fp',
                                       [num_gbboxes, tp, fp]):
        num_gbboxes = tf.cast(num_gbboxes, tf.int32)
        tp = tf.cast(tp, tf.bool)
        fp = tf.cast(fp, tf.bool)
        # Reshape TP and FP tensors and clean away 0 class values.
        tp = tf.reshape(tp, [-1])
        fp = tf.reshape(fp, [-1])

        # Local variables accumlating information over batches.
        v_num_objects = _create_local('v_num_gbboxes', shape=[], dtype=tf.int32)
        v_tp = _create_local('v_tp', shape=[0, ], dtype=tf.bool)
        v_fp = _create_local('v_fp', shape=[0, ], dtype=tf.bool)
        

        # Update operations.
        num_objects_op = state_ops.assign_add(v_num_objects,
                                           tf.reduce_sum(num_gbboxes))
        tp_op = state_ops.assign(v_tp, tf.concat([v_tp, tp], axis=0),
                                 validate_shape=False)
        fp_op = state_ops.assign(v_fp, tf.concat([v_fp, fp], axis=0),
                                 validate_shape=False)

        # Value and update ops.
        val = (v_num_objects, v_tp, v_fp)
        with ops.control_dependencies([num_objects_op, tp_op, fp_op]):
            update_op = (num_objects_op, tp_op, fp_op)

        return val, update_op


def precision_recall(num_gbboxes, tp, fp, scope=None):
    """Compute precision and recall from true positives and false
    positives booleans arrays
    """

    # Sort by score.
    with tf.name_scope(scope, 'precision_recall'):
        # Computer recall and precision.
        tp = tf.reduce_sum(tf.cast(tp, tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast(fp, tf.float32), axis=0)
        recall = tfe_math.safe_divide(tp, tf.cast(num_gbboxes, tf.float32), 'recall')
        precision = tfe_math.safe_divide(tp, tp + fp, 'precision')
        return tf.tuple([precision, recall])
    
def fmean(pre, rec):
    """Compute f-mean with precision and recall
    """
    def zero():
        return tf.zeros([])
    def not_zero():
        return 2 * pre * rec / (pre + rec)
    
    return tf.cond(pre + rec > 0, not_zero, zero)
