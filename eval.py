from train.train_evaluator import *
from model.eval_model import OneLinearLayer
from model.base import Predictor
from util.vis import *
from util.utils import *
from util.encode import *


def z_evaluator(args, model, xy_clf, device, eval_model, model_name, trainset=None, testset=None):
    # ----------------------------------------------------- encoding step
    if model_name == 'ours':
        if args.data_name != 'yaleb':
            representation_tr, _, _, _ = encode_all(args, trainset, model, xy_clf, device, is_train=True)
            representation_te, _, _, _ = encode_all(args, testset, model, xy_clf, device, is_train=False)
        else:
            representation_tr, _, s_tr, y_tr = encode_all_yaleb(args, model, xy_clf, device, is_train=True, data=trainset)
            representation_te, _, s_te, y_te = encode_all_yaleb(args, model, xy_clf, device, is_train=False, data=testset)
    elif model_name == 'odfr':
        if args.data_name != 'yaleb':
            representation_tr, _, s_tr, y_tr = odfr_encode(args, trainset, model, device)
            representation_te, _, s_te, y_te = odfr_encode(args, testset, model, device)
        else:
            representation_tr, _, s_tr, y_tr = odfr_encode_yaleb(args, model, device, is_train=True, data=trainset)
            representation_te, _, s_te, y_te = odfr_encode_yaleb(args, model, device, is_train=False, data=trainset)
    elif model_name == 'maxent':  # CI, MaxEnt
        if args.data_name != 'yaleb':
            representation_tr, s_tr, y_tr = maxent_encode(args, trainset, model, device)
            representation_te, s_te, y_te = maxent_encode(args, testset, model, device)
        else:
            representation_tr, s_tr, y_tr = maxent_encode_yaleb(args, model, device, is_train=True, data=trainset)
            representation_te, s_te, y_te = maxent_encode_yaleb(args, model, device, is_train=False, data=trainset)

    # ----------------------------------- 'tabular'
    if args.data_name != 'yaleb': 
        tr_dl, te_dl = get_representation_loader(representation_tr, representation_te, trainset, testset, args.batch_size_te)
        print(f'len train loader :{len(tr_dl)}')  # should be eq with fair epochs
        print(f'len test loader :{len(te_dl)}')

        y_predictor = train_evalutator(args, 'y', tr_dl, te_dl, device, eval_model)
        y_pred, y_acc = evaluator_predict(y_predictor, te_dl, 'y', device)
        
        # s predictor : always mlp
        s_predictor = train_evalutator(args, 's', tr_dl, te_dl, device, eval_model='mlp')
        s_acc = evaluator_predict(s_predictor, te_dl, 's', device)
        return y_pred, y_acc, s_acc, None
    # ----------------------------------- 'yaleb'
    else:
        clf_y = OneLinearLayer(args.latent_dim, args.y_dim) if eval_model == "lr" else Predictor(args.latent_dim, args.y_dim, args.hidden_units, args)
        clf_s = Predictor(args.latent_dim, args.s_dim, args.hidden_units, args)

        clf_y = train_yaleb_evaluator(args, torch.tensor(representation_tr), torch.tensor(y_tr), torch.tensor(representation_te), torch.tensor(y_te), clf_y, device, flag='y', eval_model=eval_model)
        y_acc = ((torch.argmax(clf_y(torch.tensor(representation_te).to(device)), dim=1).detach().cpu() == torch.tensor(y_te)).sum() / y_te.shape[0]).item()
        
        # s predictor
        clf_s = train_yaleb_evaluator(args, torch.tensor(representation_tr), torch.tensor(s_tr), torch.tensor(representation_te), torch.tensor(s_te), clf_s, device, flag='s', eval_model='mlp')
        s_pred = torch.argmax(clf_s(torch.tensor(representation_te).to(device)), dim=1).detach().cpu()
        s_acc = ((s_pred == torch.tensor(s_te)).sum() / s_te.shape[0]).item()
        return None, y_acc, s_acc, s_pred


def evaluator_predict(predictor, test_dl, target, device):
    predictor.eval()
    N_test, correct_cnt = 0, 0
    y_pred_raw = torch.tensor([], device=device)
    with torch.no_grad():
        for feature, s, y in test_dl:
            feature, s, y = feature.to(device), s.to(device), y.to(device)
            prediction_logit = predictor(feature)

            # get accuracy
            N_test += feature.size(0)
            if target == 'y':
                correct_cnt += ((prediction_logit > 0) == y).sum().item()
                y_pred_raw = torch.cat((y_pred_raw, prediction_logit))
            else:
                correct_cnt += ((prediction_logit > 0) == s).sum().item()

    if target == 'y':
        mask = y_pred_raw < 0
        y_pred = torch.ones_like(y_pred_raw)
        y_pred[mask] = 0
        return (y_pred.cpu()).reshape(-1, 1), correct_cnt / N_test  # y pred, y acc
    else:
        return correct_cnt / N_test  # s acc


# -----------------------------------
# Fairness Metrics
# -----------------------------------
def TPR_gap(x_test, pred):
    '''the difference in TPR between sensitive att for each main task class'''
    g11 = np.logical_and((x_test.iloc[:, 0] == 0), (x_test.iloc[:, -1] == 1))
    g21 = np.logical_and((x_test.iloc[:, 0] != 0), (x_test.iloc[:, -1] == 1))
    return torch.abs(pred[g11].mean() - pred[g21].mean())


def DP(x_test, pred):
    '''demographic parity distance
        pred : model prediction label (0, 1)
    '''
    g1 = x_test.iloc[:, 0] == 0  # group 1 bool vector
    g2 = x_test.iloc[:, 0] != 0  # group 2 bool vector
    return torch.abs(pred[g1].sum()/len(g1) - pred[g2].sum()/len(g2))


def EOpp1(x_test, pred):
    '''target - advantage : 1 / disadvantage : 0'''
    g1a = np.logical_and((x_test.iloc[:, 0] == 0), (x_test.iloc[:, -1] == 1))  # group 1, adv bool vector
    g2a = np.logical_and((x_test.iloc[:, 0] == 1), (x_test.iloc[:, -1] == 1))  # group 2, adv bool vector
    return torch.abs(pred[g1a].sum()/len(g1a) - pred[g2a].sum()/len(g2a))


def EOpp0(x_test, pred):
    '''target - advantage : 1 / disadvantage : 0'''
    g1da = np.logical_and((x_test.iloc[:, 0] == 0), (x_test.iloc[:, -1] == 0))  # group 1, disadv bool vector
    g2da = np.logical_and((x_test.iloc[:, 0] == 1), (x_test.iloc[:, -1] == 0))  # group 2, disadv bool vector
    return torch.abs(pred[g1da].sum()/len(g1da) - pred[g2da].sum()/len(g2da))


def metric_scores(test_file, pred):
    x_test = pd.read_csv(test_file)
    dp = DP(x_test, pred).item()
    gap = TPR_gap(x_test, pred).item()
    eopp1 = EOpp1(x_test, pred).item()
    eopp0 = EOpp0(x_test, pred).item()
    eodd = 0.5 * (eopp1 + eopp0)
    return dp, eopp1, eopp0, eodd, gap