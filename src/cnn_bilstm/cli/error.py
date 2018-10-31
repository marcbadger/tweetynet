import os
import pickle
import sys
from configparser import ConfigParser, NoOptionError
from glob import glob

import joblib
import numpy as np
import tensorflow as tf


from .. import metrics, utils
from ..model import CNNBiLSTM


def error(config_file):
    """measure error of a single trained model on specified data set(s)"""
    if not config_file.endswith('.ini'):
        raise ValueError(
            '{} is not a valid config file, must have .ini extension'
            .format(config_file))
    config = ConfigParser()
    config.read(config_file)

    normalize_spectrograms = config.getboolean('TRAIN',
                                               'normalize_spectrograms')


    if (not config.has_option('ERROR', 'train_data_path') and
            not config.has_option('ERROR', 'test_data_path')):
        raise NoOptionError("Must specify at least one of 'train_data_path' "
                            "and 'test_data_path' in [ERROR] section of "
                            "config.ini")

    if config.has_option('ERROR', 'train_data_path'):
        train_data_dict_path = config['TRAIN']['train_data_path']
        train_data_dict = joblib.load(train_data_dict_path)
        # load X train below in loop
        (Y_train,
         train_timebin_dur,
         train_spect_params,
         train_labels) = (train_data_dict['Y_train'],
                          train_data_dict['timebin_dur'],
                          train_data_dict['spect_params'],
                          train_data_dict['labels'])
        labels_mapping = train_data_dict['labels_mapping']
        n_syllables = len(labels_mapping)

        train_inds_file = \
            glob(os.path.join(training_records_dir, 'train_inds'))[0]
        with open(os.path.join(train_inds_file), 'rb') as train_inds_file:
            train_inds = pickle.load(train_inds_file)

        # get training set
        Y_train_subset = Y_train[train_inds]
        X_train_subset = joblib.load(os.path.join(
            training_records_dir,
            'scaled_spects_duration_{}_replicate_{}'.format(
                train_set_dur, replicate)
        ))['X_train_subset_scaled']
        assert Y_train_subset.shape[0] == X_train_subset.shape[0], \
            "mismatch between X and Y train subset shapes"


    if config.has_option('ERROR', 'test_data_path'):
        print('loading testing data')
        test_data_dict_path = config['ERROR']['test_data_path']
        test_data_dict = joblib.load(test_data_dict_path)

        # notice data is called `X_test_copy` and `Y_test_copy`
        # because main loop below needs a copy of the original
        # to normalize and reshape
        (X_test_copy,
         Y_test_copy,
         test_timebin_dur,
         files_used,
         test_spect_params,
         test_labels) = (test_data_dict['X_test'],
                         test_data_dict['Y_test'],
                         test_data_dict['timebin_dur'],
                         test_data_dict['filenames'],
                         test_data_dict['spect_params'],
                         test_data_dict['labels'])
        # have to transpose X_test
        # so rows are timebins and columns are frequencies
        X_test_copy = X_test_copy.T


    if 'train_spect_params' in locals() and 'test_spect_params' in locals():
        assert train_spect_params == test_spect_params
        assert train_timebin_dur == test_timebin_dur

    # initialize arrays to hold summary results
    if config.has_option('ERROR', 'train_data_path'):
        Y_pred_train_all = []  # will be a nested list
        Y_pred_train_labels_all = []
        train_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
        train_lev_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
        train_syl_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))

    if config.has_option('ERROR', 'test_data_path'):
        Y_pred_test_all = []  # will be a nested list
        Y_pred_test_labels_all = []
        test_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
        test_lev_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
        test_syl_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))


    # Normalize before reshaping to avoid even more convoluted array reshaping.
    # Train spectrograms were already normalized
    # just need to normalize test spects
    if normalize_spectrograms:
        scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                       .format(train_set_dur, replicate))
        spect_scaler = joblib.load(
            os.path.join(results_dirname, scaler_name))
        X_test = spect_scaler.transform(X_test_copy)
    else:
        # get back "un-reshaped" X_test
        X_test = np.copy(X_test_copy)

    # need to get Y_test from copy
    # because it gets reshaped every time through loop
    Y_test = np.copy(Y_test_copy)


    batch_size = int(config['NETWORK']['batch_size'])
    time_steps = int(config['NETWORK']['time_steps'])
    n_syllables = int(config['NETWORK']['n_syllables'])
    input_vec_size = int(config['NETWORK']['input_vec_size'])


    # now that we normalized, we can reshape
    (X_train_subset,
     Y_train_subset,
     num_batches_train) = utils.reshape_data_for_batching(
        X_train_subset,
        Y_train_subset,
        batch_size,
        time_steps,
        input_vec_size)

    (X_test,
     Y_test,
     num_batches_test) = utils.reshape_data_for_batching(
        X_test,
        Y_test,
        batch_size,
        time_steps,
        input_vec_size)

    model = CNNBiLSTM(n_syllables=n_syllables,
        input_vec_size=input_vec_size,
        batch_size=batch_size)

    with tf.Session(graph=model.graph) as sess:
        tf.logging.set_verbosity(tf.logging.ERROR)

        checkpoint_dir = config['PREDICT']['checkpoint_dir']
        meta_file = glob(os.path.join(checkpoint_dir,
                                      'checkpoint*meta*'))
        if len(meta_file) > 1:
            raise ValueError('found more than one .meta file in {}'
                             .format(checkpoint_dir))
        else:
            meta_file = meta_file[0]

        data_file = glob(os.path.join(checkpoint_dir,
                                      'checkpoint*data*'))
        if len(data_file) > 1:
            raise ValueError('found more than one .data file in {}'
                             .format(checkpoint_dir))
        else:
            data_file = data_file[0]

        model.restore(sess=sess,
        meta_file=meta_file,
        data_file=data_file)

        if config.has_option('ERROR', 'train_data_path'):
            print('calculating training set error')
            for b in range(num_batches_train):  # "b" is "batch number"
                d = {model.X: X_train_subset[:,
                              b * time_steps: (b + 1) * time_steps, :],
                     model.lng: [time_steps] * batch_size}

                if 'Y_pred_train' in locals():
                    preds = sess.run(model.predict, feed_dict=d)
                    preds = preds.reshape(batch_size, -1)
                    Y_pred_train = np.concatenate((Y_pred_train, preds),
                                                  axis=1)
                else:
                    Y_pred_train = sess.run(model.predict, feed_dict=d)
                    Y_pred_train = Y_pred_train.reshape(batch_size, -1)

            Y_train_subset = Y_train[
                train_inds]  # get back "unreshaped" Y_train_subset
            # get rid of predictions to zero padding that don't matter
            Y_pred_train = Y_pred_train.ravel()[:Y_train_subset.shape[0],
                           np.newaxis]
            train_err = np.sum(Y_pred_train - Y_train_subset != 0) / \
                        Y_train_subset.shape[0]
            train_err_arr[dur_ind, rep_ind] = train_err
            print('train error was {}'.format(train_err))
            Y_pred_train_this_dur.append(Y_pred_train)

            Y_train_subset_labels = utils.convert_timebins_to_labels(
                Y_train_subset,
                labels_mapping)
            Y_pred_train_labels = utils.convert_timebins_to_labels(
                Y_pred_train,
                labels_mapping)
            Y_pred_train_labels_this_dur.append(Y_pred_train_labels)

            if all([type(el) is int for el in Y_train_subset_labels]):
                # if labels are ints instead of str
                # convert to str just to calculate Levenshtein distance
                # and syllable error rate.
                # Let them be weird characters (e.g. '\t') because that doesn't matter
                # for calculating Levenshtein distance / syl err rate
                Y_train_subset_labels = ''.join(
                    [chr(el) for el in Y_train_subset_labels])
                Y_pred_train_labels = ''.join(
                    [chr(el) for el in Y_pred_train_labels])

            train_lev = metrics.levenshtein(Y_pred_train_labels,
                                                       Y_train_subset_labels)
            train_lev_arr[dur_ind, rep_ind] = train_lev
            print('Levenshtein distance for train set was {}'.format(
                train_lev))
            train_syl_err_rate = metrics.syllable_error_rate(
                Y_train_subset_labels,
                Y_pred_train_labels)
            train_syl_err_arr[dur_ind, rep_ind] = train_syl_err_rate
            print('Syllable error rate for train set was {}'.format(
                train_syl_err_rate))

        if config.has_option('ERROR', 'train_data_path'):
            print('calculating test set error')
            for b in range(num_batches_test):  # "b" is "batch number"
                d = {
                    model.X: X_test[:, b * time_steps: (b + 1) * time_steps,
                             :],
                    model.lng: [time_steps] * batch_size}

                if 'Y_pred_test' in locals():
                    preds = sess.run(model.predict, feed_dict=d)
                    preds = preds.reshape(batch_size, -1)
                    Y_pred_test = np.concatenate((Y_pred_test, preds),
                                                 axis=1)
                else:
                    Y_pred_test = sess.run(model.predict, feed_dict=d)
                    Y_pred_test = Y_pred_test.reshape(batch_size, -1)

            # again get rid of zero padding predictions
            Y_pred_test = Y_pred_test.ravel()[:Y_test_copy.shape[0],
                          np.newaxis]
            test_err = np.sum(Y_pred_test - Y_test_copy != 0) / \
                       Y_test_copy.shape[0]
            test_err_arr[dur_ind, rep_ind] = test_err
            print('test error was {}'.format(test_err))
            Y_pred_test_this_dur.append(Y_pred_test)

            Y_pred_test_labels = utils.convert_timebins_to_labels(
                Y_pred_test,
                labels_mapping)
            Y_pred_test_labels_this_dur.append(Y_pred_test_labels)
            if all([type(el) is int for el in Y_pred_test_labels]):
                # if labels are ints instead of str
                # convert to str just to calculate Levenshtein distance
                # and syllable error rate.
                # Let them be weird characters (e.g. '\t') because that doesn't matter
                # for calculating Levenshtein distance / syl err rate
                Y_pred_test_labels = ''.join(
                    [chr(el) for el in Y_pred_test_labels])
                # already converted actual Y_test_labels from int to str above,
                # stored in variable `Y_test_labels_for_lev`

            test_lev = metrics.levenshtein(Y_pred_test_labels,
                                                      Y_test_labels_for_lev)
            test_lev_arr[dur_ind, rep_ind] = test_lev
            print(
                'Levenshtein distance for test set was {}'.format(test_lev))
            test_syl_err_rate = metrics.syllable_error_rate(
                Y_test_labels_for_lev,
                Y_pred_test_labels)
            print('Syllable error rate for test set was {}'.format(
                test_syl_err_rate))
            test_syl_err_arr[dur_ind, rep_ind] = test_syl_err_rate

    Y_pred_train_all.append(Y_pred_train_this_dur)
    Y_pred_test_all.append(Y_pred_test_this_dur)
    Y_pred_train_labels_all.append(Y_pred_train_labels_this_dur)
    Y_pred_test_labels_all.append(Y_pred_test_labels_this_dur)

    Y_pred_train_filename = os.path.join(summary_dirname,
                                         'Y_pred_train_all')
    with open(Y_pred_train_filename, 'wb') as Y_pred_train_file:
        pickle.dump(Y_pred_train_all, Y_pred_train_file)

    Y_pred_test_filename = os.path.join(summary_dirname,
                                        'Y_pred_test_all')
    with open(Y_pred_test_filename, 'wb') as Y_pred_test_file:
        pickle.dump(Y_pred_test_all, Y_pred_test_file)

    train_err_filename = os.path.join(summary_dirname,
                                      'train_err')
    with open(train_err_filename, 'wb') as train_err_file:
        pickle.dump(train_err_arr, train_err_file)

    test_err_filename = os.path.join(summary_dirname,
                                     'test_err')
    with open(test_err_filename, 'wb') as test_err_file:
        pickle.dump(test_err_arr, test_err_file)

    pred_and_err_dict = {'Y_pred_train_all': Y_pred_train_all,
                         'Y_pred_test_all': Y_pred_test_all,
                         'Y_pred_train_labels_all': Y_pred_train_labels_all,
                         'Y_pred_test_labels_all': Y_pred_test_labels_all,
                         'Y_train_labels': Y_train_labels,
                         'Y_test_labels': Y_test_labels,
                         'train_err': train_err_arr,
                         'test_err': test_err_arr,
                         'train_lev': train_lev_arr,
                         'train_syl_err_rate': train_syl_err_arr,
                         'test_lev': test_lev_arr,
                         'test_syl_err_rate': test_syl_err_arr,
                         'train_set_durs': TRAIN_SET_DURS}

    pred_err_dict_filename = os.path.join(summary_dirname,
                                          'y_preds_and_err_for_train_and_test')
    joblib.dump(pred_and_err_dict, pred_err_dict_filename)

if __name__ == '__main__':
    config_file = sys.argv[1]
    error(config_file)
