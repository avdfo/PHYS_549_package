import os, inspect, time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def makedir(path):
    try: os.mkdir(path)
    except: pass

def save_graph(contents, xlabel, ylabel, savename):
    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def save_graph_2(contents_1, contents_2, xlabel, ylabel, savename):
    np.save(savename, np.asarray(contents_1))
    np.save(savename + '_val', np.asarray(contents_2))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents_1, color='blue', linestyle="-", label="Training loss")
    plt.plot(contents_2, color='orange', linestyle="-", label="Validation loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def save_graph_3(contents_1, contents_2, xlabel, ylabel, savename):
    np.save(savename, np.asarray(contents_1))
    np.save(savename + '_val', np.asarray(contents_2))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents_1, color='blue', linestyle="-", label="Training PSNR")
    plt.plot(contents_2, color='orange', linestyle="-", label="Validation PSNR")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def training(sess, neuralnet, saver, dataset, epochs, batch_size, batch_size_val):

    start_time = time.time()
    loss_tr = 0

    list_loss = []
    list_loss_val = []
    epoch_loss_tr = []
    epoch_loss_val = []

    list_psnr = []
    list_psnr_val = []
    epoch_psnr_tr = []
    epoch_psnr_val = []

    list_psnr_static = []

    makedir(PACK_PATH+"/training")
    makedir(PACK_PATH+"/static")
    makedir(PACK_PATH+"/static/reconstruction")

    if (os.path.exists(PACK_PATH + "/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH + "/Checkpoint/model_checker")

    print("\nTraining SRCNN to %d epochs" %(epochs))
    train_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint')

    X_static, Y_static, _ = dataset.next_train(batch_size=1)
    img_input = np.squeeze(X_static, axis=0)
    img_ground = np.squeeze(Y_static, axis=0)
    img_input = np.squeeze(img_input, axis=2)
    img_ground = np.squeeze(img_ground, axis=2)
    plt.imsave("%s/static/low-resolution.png" %(PACK_PATH), img_input)
    plt.imsave("%s/static/high-resolution.png" %(PACK_PATH), img_ground)

    iteration = 0
    for epoch in range(epochs):
        dataset.list_train_lr, dataset.list_train_hr = shuffle(dataset.list_train_lr, dataset.list_train_hr)
        dataset.list_validation_lr, dataset.list_validation_hr = shuffle(dataset.list_validation_lr, dataset.list_validation_hr)
        while(True):
            X_tr, Y_tr, terminator = dataset.next_train(batch_size=batch_size)
            summaries, _ = sess.run([neuralnet.summaries, neuralnet.optimizer], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            loss_tr, psnr_tr = sess.run([neuralnet.loss, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
            list_loss.append(loss_tr)
            list_psnr.append(psnr_tr)
            train_writer.add_summary(summaries, iteration)

            X_val, Y_val = dataset.next_val(batch_size_val=batch_size_val)
            loss_val, psnr_val = sess.run([neuralnet.loss, neuralnet.psnr], feed_dict={neuralnet.inputs:X_val, neuralnet.outputs: Y_val})
            loss_val = loss_val / batch_size_val * batch_size
            list_loss_val.append(loss_val)
            list_psnr_val.append(psnr_val)
            iteration += 1

            if(terminator):
                dataset.idx_val = 0
                break

        # Get the first image in the batch.
        X_tmp, Y_tmp = np.expand_dims(X_tr[0], axis=0), np.expand_dims(Y_tr[0], axis=0)

        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tmp, neuralnet.outputs:Y_tmp})
        img_input, img_recon, img_ground = np.squeeze(X_tmp, axis=0), np.squeeze(img_recon, axis=0), np.squeeze(Y_tmp, axis=0)

        img_input = np.squeeze(img_input, axis=2)
        img_recon = np.squeeze(img_recon, axis=2)
        img_ground = np.squeeze(img_ground, axis=2)

        plt.clf()
        plt.rcParams['font.size'] = 100
        plt.figure(figsize=(100, 40))
        plt.subplot(131)
        plt.title("Low-Resolution")
        plt.imshow(img_input)
        plt.subplot(132)
        plt.title("Reconstruction")
        plt.imshow(img_recon)
        plt.subplot(133)
        plt.title("High-Resolution")
        plt.imshow(img_ground)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.savefig("%s/training/%09d_psnr_%d.png" %(PACK_PATH, epoch, int(tmp_psnr)))
        plt.close()

        """static img(test)"""
        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_static, neuralnet.outputs:Y_static})
        list_psnr_static.append(tmp_psnr)
        img_recon = np.squeeze(img_recon, axis=0)
        img_recon = np.squeeze(img_recon, axis=2)
        plt.imsave("%s/static/reconstruction/%09d_psnr_%d.png" %(PACK_PATH, epoch, int(tmp_psnr)), img_recon)

        epoch_loss_tr.append(loss_tr)
        epoch_loss_val.append(loss_val)
        epoch_psnr_tr.append(psnr_tr)
        epoch_psnr_val.append(psnr_val)
        print("Epoch [%d / %d] | Loss: %f  Val_loss: %f  PSNR: %f  Val_PSNR: %f" %(epoch+1, epochs, loss_tr, loss_val, psnr_tr, psnr_val))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("Final Epcoh | Loss: %f  PSNR: %f" %(loss_tr, psnr_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph_2(contents_1=list_loss, contents_2=list_loss_val, xlabel="Number of backprops", ylabel="Cross entropy loss", savename="loss")
    save_graph_3(contents_1=list_psnr, contents_2=list_psnr_val, xlabel="Number of backprops", ylabel="PSNR (dB)", savename="psnr")
    save_graph_2(contents_1=epoch_loss_tr, contents_2=epoch_loss_val, xlabel="Epochs",
                 ylabel="Cross entropy loss", savename="loss_epoch")
    save_graph_3(contents_1=epoch_psnr_tr, contents_2=epoch_psnr_val, xlabel="Epochs", ylabel="PSNR (dB)",
                 savename="psnr_epoch")
    save_graph(contents=list_psnr_static, xlabel="Number of backprops", ylabel="PSNR (dB)", savename="psnr_static")

def testing(sess, neuralnet, saver, dataset):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    makedir(PACK_PATH+"/test")
    makedir(PACK_PATH+"/test/reconstruction")

    start_time = time.time()
    print("\nTesting")
    for tidx in range(dataset.amount_te):

        X_te, Y_te = dataset.next_test()
        if(X_te is None): break

        img_recon, tmp_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_te, neuralnet.outputs:Y_te})
        img_recon = np.squeeze(img_recon, axis=0)
        img_recon = np.squeeze(img_recon, axis=2)
        np.savetxt("%s/test/reconstruction/%09d_psnr_%d.txt" %(PACK_PATH, tidx, int(tmp_psnr)), img_recon)
        plt.imsave("%s/test/reconstruction/%09d_psnr_%d.png" %(PACK_PATH, tidx, int(tmp_psnr)), img_recon)

        img_input = np.squeeze(X_te, axis=0)
        img_input = np.squeeze(img_input, axis=2)
        img_ground = np.squeeze(Y_te, axis=0)
        img_ground = np.squeeze(img_ground, axis=2)
        plt.imsave("%s/test/low-resolution.png" %(PACK_PATH), img_input)
        plt.imsave("%s/test/high-resolution.png" %(PACK_PATH), img_ground)

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

def testing_exp(sess, neuralnet, saver, dataset):
    if (os.path.exists(PACK_PATH + "/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH + "/Checkpoint/model_checker")

    makedir(PACK_PATH + "/test_exp")
    makedir(PACK_PATH + "/test_exp/original")
    makedir(PACK_PATH + "/test_exp/reconstruction")

    start_time = time.time()
    print("\nTesting experimental data")
    for i_exp in range(dataset.amount_te_exp):
        X_exp = dataset.next_test_exp()
        if (X_exp is None): break

        img_recon = sess.run(neuralnet.recon, feed_dict={neuralnet.inputs:X_exp})
        img_recon = np.squeeze(img_recon, axis=0)
        img_recon = np.squeeze(img_recon, axis=2)

        np.savetxt("%s/test_exp/reconstruction/%04d.txt" %(PACK_PATH, i_exp), img_recon)
        plt.imsave("%s/test_exp/reconstruction/%04d.png" % (PACK_PATH, i_exp), img_recon)

        img_input = np.squeeze(X_exp, axis=0)
        img_input = np.squeeze(img_input, axis=2)
        plt.imsave("%s/test_exp/original/%04d.png" %(PACK_PATH, i_exp), img_input)

    elapsed_time = time.time() - start_time
    print("Elapsed: " + str(elapsed_time))