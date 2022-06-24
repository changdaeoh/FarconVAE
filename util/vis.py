import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-white")
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
#import wandb


def vis_z_yaleb(args, z_s, z_non_s, sense, y, saving_name, is_train=False):
    model_name = args.model_name if args.model_name != 'farcon' else args.model_name + '_' + str(args.cont_loss_ver)
    target = pd.Series(y, name='target')
    if args.model_name not in ["odfr", "ours"]:  # CI, MaxEnt
        # sensitive non-related representation
        z_non_df = pd.DataFrame(z_non_s)
        z_non_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_non_df)
        non_tsne_df = pd.DataFrame(z_non_tsne, columns=['dim1', 'dim2'])
        vis_non_frame = pd.concat([sense, non_tsne_df, target], axis=1)

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='sensitive', palette=['#b25da6', '#6688c3', '#48a56a', '#eaaf41', '#ce4a4a'], s=18)
        plt.legend(title=r"$S$", title_fontsize=12, fontsize=9)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_s.png', dpi=285)
        plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='target', palette='Spectral', legend = False, s=18)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_y.png', dpi=285)

    else:
        n = z_non_s.shape[0]
        dummy_0 = np.zeros((n, 1))
        dummy_1 = np.ones((n, 1))
        z_label = pd.DataFrame(np.concatenate((dummy_0, dummy_1), axis=0), columns=['is_zx'])
        sense_2n = pd.concat((sense, sense), axis=0).reset_index(drop=True)
        target_2n = pd.concat((target, target), axis=0).reset_index(drop=True)

        # 2N * latent dim
        z_concat = pd.DataFrame(np.concatenate((z_s, z_non_s), axis=0))
        z_concat_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_concat)
        z_cont_df = pd.DataFrame(z_concat_tsne, columns=['dim1', 'dim2'])
        cont_frame = pd.concat((z_cont_df, z_label, sense_2n, target_2n), axis=1)

        # sense related representation
        z_s_df = pd.DataFrame(z_s)
        z_s_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_s_df)
        s_tsne_df = pd.DataFrame(z_s_tsne, columns=['dim1', 'dim2'])
        z_s_frame = pd.concat([sense, s_tsne_df, target], axis=1)

        # sense non-related representation
        z_non_df = pd.DataFrame(z_non_s)
        z_non_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_non_df)
        non_tsne_df = pd.DataFrame(z_non_tsne, columns=['dim1', 'dim2'])
        vis_non_frame = pd.concat([sense, non_tsne_df, target], axis=1)

        cont_frame.loc[cont_frame['is_zx'] == 1, 'Representation'] = 'Zx'
        cont_frame.loc[cont_frame['is_zx'] == 0, 'Representation'] = 'Zs'
        z_s_frame.rename(columns={'sensitive':'S'}, inplace=True)
        vis_non_frame.rename(columns={'sensitive':'S'}, inplace=True)
        cont_frame.rename(columns={'sensitive':'S'}, inplace=True)
        cont_frame['dim1'] = pd.to_numeric(cont_frame["dim1"])
        cont_frame['dim1'] = pd.to_numeric(cont_frame["dim1"])

        # visualize
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='S', palette=['#b25da6', '#6688c3', '#48a56a', '#eaaf41', '#ce4a4a'], s=18)
        plt.legend(title=r"$S$", title_fontsize=12, fontsize=9)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(f'{args.vis_path + args.data_name}_{model_name}_zx_s.png', dpi=285)
        plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=z_s_frame, x='dim1', y='dim2', hue='S', palette=['#b25da6', '#6688c3', '#48a56a', '#eaaf41', '#ce4a4a'], s=18)
        plt.legend(title=r"$S$", title_fontsize=12, fontsize=9)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zs_s.png', dpi=285)
        plt.clf()

        # plt.figure(figsize=(6, 6))
        # sns.scatterplot(data=cont_frame, x='dim1', y='dim2', style='Representation', hue='S', palette=['#b25da6', '#6688c3', '#48a56a', '#eaaf41', '#ce4a4a'], s=15)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left',borderpad = 1)
        # plt.xticks([])
        # plt.yticks([])
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zxs_s.png', dpi=285)

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='target', palette='Spectral', legend = False, s=18)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zx_y.png', dpi=285)
        plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=z_s_frame, x='dim1', y='dim2', hue='target', palette='Spectral', legend = False, s=18)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zs_y.png', dpi=285)
        plt.clf()

        # plt.figure(figsize=(6, 6))
        # sns.scatterplot(data=cont_frame, x='dim1', y='dim2', style='is_zx', hue='target', palette='Spectral', legend = False, s=18)
        # plt.xticks([])
        # plt.yticks([])
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zxs_y.png', dpi=285)


def vis_z_binary(args, z_s, z_non_s, sense, target, z_vis_name, is_train=True):
    model_name = args.model_name if args.model_name != 'farcon' else args.model_name + '_' + str(args.cont_loss_ver)
    if args.model_name not in ["odfr", "ours"]:
        if z_non_s.shape[1] > 2:
            z_non_df = pd.DataFrame(z_non_s)
            z_non_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_non_df)
            non_tsne_df = pd.DataFrame(z_non_tsne, columns=['dim1', 'dim2'])
        else:
            non_tsne_df = pd.DataFrame(z_non_s, columns=['dim1', 'dim2'])
        vis_non_frame = pd.concat([sense, non_tsne_df, target], axis=1)

        vis_non_frame.loc[vis_non_frame[args.sensitive] == 1, 'S'] = 'Male'
        vis_non_frame.loc[vis_non_frame[args.sensitive] == 0, 'S'] = 'Female'
        vis_non_frame.loc[vis_non_frame[args.target] == 1, 'Y'] = 'Bad Credit' if args.data_name == 'german' else 'High Income'
        vis_non_frame.loc[vis_non_frame[args.target] == 0, 'Y'] = 'Good Credit' if args.data_name == 'german' else 'Low Income'

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='S', palette=['#F8766D', '#02BEC3'], s=18)
        plt.legend(title=r"$S$", title_fontsize=12, fontsize=9)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_s.png', dpi=285)
        plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='Y', palette=['#4A36F5', '#FFC820'], s=18)
        plt.legend(title=r"$Y$", title_fontsize=12, fontsize=9)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_y.png', dpi=285)

    else:
        n = z_non_s.shape[0]
        dummy_0 = np.zeros((n, 1))
        dummy_1 = np.ones((n, 1))
        z_label = pd.DataFrame(np.concatenate((dummy_0, dummy_1), axis=0), columns=['is_zx'])
        sense_2n = pd.concat((sense, sense), axis=0).reset_index(drop=True)
        target_2n = pd.concat((target, target), axis=0).reset_index(drop=True)

        # 2N * latent dim
        if z_non_s.shape[1] > 2:
            z_concat = pd.DataFrame(np.concatenate((z_s, z_non_s), axis=0))
            z_concat_emd = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_concat)
            z_cont_df = pd.DataFrame(z_concat_emd, columns=['dim1', 'dim2'])
        else:
            z_cont_df = pd.DataFrame(np.concatenate((z_s, z_non_s), axis=0), columns=['dim1', 'dim2'])
        cont_frame = pd.concat((z_cont_df, z_label, sense_2n, target_2n), axis=1)

        # sensitive related representation
        if z_non_s.shape[1] > 2:
            z_s_df = pd.DataFrame(z_s)
            z_s_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_s_df)
            s_tsne_df = pd.DataFrame(z_s_tsne, columns=['dim1', 'dim2'])
        else:
            s_tsne_df = pd.DataFrame(z_s, columns=['dim1', 'dim2'])
        z_s_frame = pd.concat([sense, s_tsne_df, target], axis=1)

        # sensitive non-related representation
        if z_non_s.shape[1] > 2:
            z_non_df = pd.DataFrame(z_non_s)
            z_non_tsne = TSNE(n_components=2, n_iter=700, perplexity=20).fit_transform(z_non_df)
            non_tsne_df = pd.DataFrame(z_non_tsne, columns=['dim1', 'dim2'])
        else:
            non_tsne_df = pd.DataFrame(z_non_s, columns=['dim1', 'dim2'])
        vis_non_frame = pd.concat([sense, non_tsne_df, target], axis=1)

        z_s_frame.loc[z_s_frame[args.sensitive] == 1, 'S'] = 'Male'
        z_s_frame.loc[z_s_frame[args.sensitive] == 0, 'S'] = 'Female'
        z_s_frame.loc[z_s_frame[args.target] == 1, 'Y'] = 'Bad Credit' if args.data_name == 'german' else 'High Income'
        z_s_frame.loc[z_s_frame[args.target] == 0, 'Y'] = 'Good Credit' if args.data_name == 'german' else 'Low Income'

        vis_non_frame.loc[vis_non_frame[args.sensitive] == 1, 'S'] = 'Male'
        vis_non_frame.loc[vis_non_frame[args.sensitive] == 0, 'S'] = 'Female'
        vis_non_frame.loc[vis_non_frame[args.target] == 1, 'Y'] = 'Bad Credit' if args.data_name == 'german' else 'High Income'
        vis_non_frame.loc[vis_non_frame[args.target] == 0, 'Y'] = 'Good Credit' if args.data_name == 'german' else 'Low Income'

        cont_frame.loc[cont_frame[args.sensitive] == 1, 'S'] = 'Male'
        cont_frame.loc[cont_frame[args.sensitive] == 0, 'S'] = 'Female'
        cont_frame.loc[cont_frame[args.target] == 1, 'Y'] = 'Bad Credit' if args.data_name == 'german' else 'High Income'
        cont_frame.loc[cont_frame[args.target] == 0, 'Y'] = 'Good Credit' if args.data_name == 'german' else 'Low Income'
        cont_frame.loc[cont_frame['is_zx'] == 1, 'Representation'] = 'Zx'
        cont_frame.loc[cont_frame['is_zx'] == 0, 'Representation'] = 'Zs'

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='S', palette=['#F8766D', '#02BEC3'], s=18)
        plt.legend(fontsize=9, title=r'$S$', title_fontsize=12)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zx_s.png', dpi=285)
        plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=z_s_frame, x='dim1', y='dim2', hue='S', palette=['#F8766D', '#02BEC3'], s=18)
        plt.legend(fontsize=9, title=r'$S$', title_fontsize=12)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zs_s.png', dpi=285)
        plt.clf()

        # plt.figure(figsize=(6, 6))
        # sns.scatterplot(data=cont_frame, x='dim1', y='dim2', style='Representation', hue='S', palette=['#F8766D', '#02BEC3'], s=18)
        # plt.legend(fontsize=9, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,borderpad = 1)
        # plt.xticks([])
        # plt.yticks([])
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zxs_s.png', dpi=285)
        # plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=vis_non_frame, x='dim1', y='dim2', hue='Y', palette=['#4A36F5', '#FFC820'], s=18)
        plt.legend(fontsize=9, title=r'$Y$', title_fontsize=12)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zx_y.png', dpi=285)
        plt.clf()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=z_s_frame, x='dim1', y='dim2', hue='Y', palette=['#4A36F5', '#FFC820'], s=18)
        plt.legend(fontsize=9, title=r'$Y$', title_fontsize=12)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zs_y.png', dpi=285)
        plt.clf()

        # plt.figure(figsize=(6, 6))
        # sns.scatterplot(data=cont_frame, x='dim1', y='dim2', style='Representation', hue='Y', palette=['#4A36F5', '#FFC820'], s=18)
        # plt.legend(fontsize=9, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,borderpad = 1)
        # plt.xticks([])
        # plt.yticks([])
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.savefig(args.vis_path + f'{args.data_name}_{model_name}_zxs_y.png', dpi=285)