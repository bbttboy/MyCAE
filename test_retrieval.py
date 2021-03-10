import numpy as np
import torch
import random
from tqdm import tqdm 
from collections import OrderedDict


def test(opt, model, testset):
    """Tests a model over the given testset."""
    # eval()表示不改变网络参数，即只执行评估
    model.eval()
    test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []
    if test_queries:
        imgs = []
        mods = []
        for t in tqdm(test_queries):
            torch.cuda.empty_cache()
            imgs += [testset.get_img(t['source_img_id'])]
            if opt.use_complete_text_query:
                mods += [t['target_caption']]
            else:
                mods += [t['mod']['str']]
            
            if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                dct_with_representations = model.compose_img_text(imgs.cuda(), mods)
                f = dct_with_representations["repres"].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)
        all_target_captions = [t['target_caption'] for t in test_queries]

        # compute all image features
        imgs = []
        for i in tqdm(range(len(testset.imgs))):
            imgs += [testset.get_img(i)]
            if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                imgs = model.extract_img_feature(imgs.cuda()).data.cpu().numpy()

                all_imgs += [imgs]
                imgs = []
        all_imgs = np.concatenate(all_imgs)
        all_captions = [img['caption'][0] for img in testset.imgs]
    
    else:
        # use training queries to approximate training retrieval performance
        imgs0 = []
        imgs = []
        mods = []
        training_approx = 9600
        for i in range(training_approx):
            torch.cuda.empty_cache()
            item = testset[i]
            imgs += [item['source_img_data']]
            if opt.use_complete_text_query:
                mods += [item['target_caption']]
            else:
                mods += [item['mod']['str']]

            if len(imgs) >= opt.batch_size or i == training_approx:
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs)
                dct_with_representations = model.compose_img_text(imgs.cuda(), mods)
                f = dct_with_representations['repres'].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
            imgs0 += [item['target_img_data']]
            if len(imgs0) >= opt.batch_size or i == training_approx:
                imgs0 = torch.stack(imgs0).float()
                imgs0 = torch.autograd.Variable(imgs0)
                imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
                all_imgs += [imgs0]
                imgs0 = []
            all_captions += [item['target_caption']]
            all_target_captions += [item['target_caption']]
        all_imgs = np.concatenate(all_imgs)
        all_queries = np.concatenate(all_queries)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            sims[i, t['source_img_id']] -= -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    nn_result = [ [all_captions[nn] for nn in nns] for nns in nn_result ]
    for k in [1, 5, 10, 50, 100]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', r)]

    return out