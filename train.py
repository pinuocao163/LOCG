import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model, MaskedNLLLoss_1
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
from transformers import RobertaTokenizer, RobertaModel

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_label_embedding(Dataset = 'IEMOCAP'):
    # 加载 RoBERTa 模型和 tokenizer
    model_name = r'E:\Code\修改\DialogueCRN-main\roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    if args.Dataset == 'IEMOCAP':
        # 目标名称列表
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    else:
        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
    # 将目标名称转换为对应的 token
    target_tokens = [tokenizer.encode(target, add_special_tokens=False) for target in target_names]

    # 使用 RoBERTa 模型将 token 转换为向量表示
    max_length = max(len(tokens) for tokens in target_tokens)
    target_vectors = []
    for tokens in target_tokens:
        padding = [0] * (max_length - len(tokens))
        input_ids = torch.LongTensor(tokens + padding).unsqueeze(0)
        with torch.no_grad():
            output = model(input_ids)[0][:, 0, :]
        target_vectors.append(output.numpy()[0])
    # 将NumPy数组转换为PyTorch张量
    target_vectors = torch.tensor(target_vectors)

    # 将张量分配给GPU设备（如果可用）
    if torch.cuda.is_available():
        target_vectors = target_vectors.cuda()
    return target_vectors


def train_or_eval_model(model, loss_function, loss_function_1, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0,
                        gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        textf, visuf, acouf, xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        # 将0替换为0，1/3/5替换为1，2替换为2
        label_polarity = torch.where((label == 0) | (label == 4) , torch.tensor(0).cuda(),
                         torch.where((label == 1) | (label == 3) | (label == 5), torch.tensor(1).cuda(),
                         torch.where(label == 2, torch.tensor(2).cuda(), label)))  # IEMOCAP


        # label_polarity = torch.where((label == 1) | (label == 4), torch.tensor(0).cuda(),
        #                              torch.where((label == 2) | (label == 3) | (label == 5) | (label == 6), torch.tensor(1).cuda(),
        #                                          torch.where(label == 0, torch.tensor(2).cuda(), label)))  # MELD


        ck = torch.stack((xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact), dim=2).transpose(0, 1)

        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        #     kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)

        all_log_prob, all_prob, softmax_similarity_onehot, all_prob_polarity = model(textf, visuf, acouf, umask, qmask, lengths, ck, label, label_embedding)

        # lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        # lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        # lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)
        label_polarity_ = label_polarity.view(-1)
        softmax_similarity_onehot_ = softmax_similarity_onehot.view(-1, softmax_similarity_onehot.size()[2])
        all_prob_polarity_ = all_prob_polarity.view(-1, all_prob_polarity.size()[2])
        # kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        # kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        # kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        # kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

        # loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
        #        gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(
        #     lp_3, labels_, umask)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3,
        #                                                                                                   kl_p_all,
        #                                                                                                   umask))
        # loss = loss_function(lp_all, labels_, umask)

        loss =gamma_1 * kl_loss(lp_all, softmax_similarity_onehot_, umask) + gamma_2 * loss_function_1(all_prob_polarity_, label_polarity_, umask) + gamma_3 * loss_function(lp_all, labels_, umask)


        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=16, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    print('temp {}'.format(args.temp))
    hidden_size = 128
    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                    n_classes=n_classes,
                                    hidden_dim=args.hidden_dim,
                                    n_speakers=n_speakers,
                                    dropout=args.dropout,
                                    hidden_size=hidden_size,
                                    )

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_weights = torch.FloatTensor(
            [1 / 0.47151867, 1 / 0.1206327, 1 / 0.02682951, 1 / 0.06837521, 1 / 0.17449194, 1 / 0.02712984,
             1 / 0.11102212])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        loss_function_1 = MaskedNLLLoss_1(None)
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        loss_function_1 = MaskedNLLLoss_1(None)

        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    label_embedding = get_label_embedding('IEMOCAP')

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, loss_function_1, kl_loss, train_loader,
                                                                           e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, loss_function_1, kl_loss, valid_loader,
                                                                           e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, loss_function_1,
                                                                                                 kl_loss, test_loader,
                                                                                                 e)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))
        if (e + 1) % 10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_ + 'record', False):
        record[key_ + 'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    else:
        record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))


