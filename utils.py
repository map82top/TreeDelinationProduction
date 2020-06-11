from skimage.measure import find_contours


def filter_prediction(prediction, min_score=0.7):
    """Delete prediction crown which has score less then less min score

    Used only prediction fron neural network
    :param prediction: dictionary
        prediction
    :param min_score: int
        crowns has score less then this value not added a new prediction
    :return:
        filtered prediction
    """
    scores = prediction['scores']
    scores_filter = prediction['scores'] > min_score
    new_scores = scores[scores_filter]
    count_allowed_levels = len(new_scores)
    new_masks = prediction['masks'][:, :, :count_allowed_levels]
    total_count_trees = 0
    for i in range(count_allowed_levels):
        total_count_trees = total_count_trees + len(find_contours(new_masks[:, :, i], i))

    new_prediction = dict()
    new_prediction['rois'] = prediction['rois'][:count_allowed_levels, :] * 10
    new_prediction['class_ids'] = prediction['class_ids'][:count_allowed_levels]
    new_prediction['scores'] = new_scores
    new_prediction['masks'] = new_masks

    return total_count_trees, new_prediction