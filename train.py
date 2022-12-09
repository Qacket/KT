from torch import optim, nn
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
from loss import VAE_Loss
from sklearn.decomposition import PCA
def train_a_vae(args, device, train_loader, a_vae):

    a_vae = a_vae.to(device)
    vae_loss = VAE_Loss()
    optimizer = optim.Adam(a_vae.parameters(), lr=args.learning_rate)


    torch.set_grad_enabled(True)
    a_vae.train()
    writer = SummaryWriter(args.annotator_writer)



    for epoch in range(args.epochs):

        for iteration, data in enumerate(train_loader):
            annotator_inputs = data[4].to(device)
            optimizer.zero_grad()

            a_output, a_mean, a_logv, a_z = a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator

            # 工人 loss
            a_KL_loss, a_recon_loss = vae_loss(mu=a_mean, log_var=a_logv, recon_x=a_output, x=annotator_inputs)

            a_mloss = a_KL_loss + a_recon_loss

            a_mloss.backward()

            optimizer.step()


        writer.add_scalar(tag='a_KL_loss', scalar_value=a_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_recon_loss', scalar_value=a_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_mloss', scalar_value=a_mloss.data.item(), global_step=epoch)
        print(epoch, a_KL_loss.data.item(), a_recon_loss.data.item(), a_mloss.data.item())
    torch.save(a_vae, args.model_dir + args.annotator_vae_name)


def train_t_vae(args, device, train_loader, t_vae):

    t_vae = t_vae.to(device)
    vae_loss = VAE_Loss()
    optimizer = optim.Adam(t_vae.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)
    t_vae.train()
    writer = SummaryWriter(args.task_writer)

    for epoch in range(args.epochs):

        for iteration, data in enumerate(train_loader):

            task_inputs = data[2].to(device)

            optimizer.zero_grad()

            t_output, t_mean, t_logv, t_z = t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task

            t_KL_loss, t_recon_loss = vae_loss(mu=t_mean, log_var=t_logv, recon_x=t_output, x=task_inputs)

            t_mloss = t_KL_loss + t_recon_loss

            t_mloss.backward()

            optimizer.step()

        writer.add_scalar(tag='t_KL_loss', scalar_value=t_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_recon_loss', scalar_value=t_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_mloss', scalar_value=t_mloss.data.item(), global_step=epoch)
        print(epoch, t_KL_loss.data.item(), t_recon_loss.data.item(), t_mloss.data.item())

    torch.save(t_vae, args.model_dir + args.task_vae_name)

def train_mymodel(args, device, train_loader, mymodel):

    mymodel = mymodel.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(mymodel.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)

    writer = SummaryWriter(args.mymodel_writer)

    for epoch in range(args.epochs):
        count = 0
        sum = 0
        mymodel.train()
        for iteration, data in enumerate(train_loader):

            annotator_inputs = data[4].to(device)
            task_inputs = data[2].to(device)
            label_tensor = torch.squeeze(data[5].type(torch.float32).type(torch.long).to(device))

            optimizer.zero_grad()

            a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
            t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task

            z = torch.cat((a_z, t_z), 1)  # z1 z2 结合

            dev_label = mymodel(z)  # 获得生成的标注^dev_label

            sup_loss = loss_fn(dev_label, label_tensor)

            # loss = t_KL_loss + t_recon_loss + sup_loss
            loss = sup_loss

            loss.backward()
            optimizer.step()


            prediction = torch.max(F.softmax(dev_label), 1)[1]
            pred_label = prediction.cpu().data.numpy().squeeze()
            target_label = label_tensor.cpu().data.numpy()

            for i in range(len(data[3])):
                if pred_label[i] == target_label[i]:
                    count += 1
            sum += len(data[3])
        acc = count / sum
        writer.add_scalar(tag='sup_loss', scalar_value=sup_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='loss', scalar_value=loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='acc', scalar_value=acc, global_step=epoch)
        print("--------------------------TRAIN----------------------------------")
        print("epoch", epoch, "train_loss", loss.data.item())
        print("train_acc:", acc)

        # mymodel.eval()
        # count = 0
        # num = 0
        # with torch.no_grad():
        #     for iteration, data in enumerate(test_loader):
        #
        #         task_id = data[1]
        #         annotator_id = data[2]
        #         label_tensor = data[3]
        #
        #         # task
        #         task_id_np = np.array(task_id).astype(dtype=int).reshape(-1, 1)
        #         task_tensor = torch.from_numpy(task_id_np)
        #         task_onehot = F.one_hot(task_tensor, 12012).resize_(len(task_tensor), 12012)
        #         task_inputs = task_onehot.type(torch.float32).to(device)
        #
        #         # annotator
        #         # featrue
        #         annotator_id_feature_np = np.array(annotator_id).astype(dtype=float).reshape(-1, 1)
        #         annotator_feature = np.repeat(annotator_id_feature_np, 1765).reshape(-1, 1765)
        #         for i in range(len(annotator_feature)):
        #             annotator_feature[i] = features[int(annotator_id_feature_np[i][0])]
        #         annotator_feature_tensor = torch.from_numpy(annotator_feature).type(torch.float32).to(device)
        #
        #         # onehot
        #         annotator_id_onehot_np = np.array(annotator_id).astype(dtype=int).reshape(-1, 1)
        #         annotator_tensor = torch.from_numpy(annotator_id_onehot_np)
        #         annotator_onehot = F.one_hot(annotator_tensor, 6000).resize_(len(annotator_tensor), 6000)
        #         annotator_onehot = annotator_onehot.type(torch.float32).to(device)
        #
        #         annotator_inputs = torch.cat((annotator_onehot, annotator_feature_tensor), 1)
        #
        #         # label
        #         label_tensor = label_tensor.type(torch.long).to(device)
        #
        #
        #         a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
        #         t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task
        #         z = torch.cat((a_z, t_z), 1)  # z1 z2 结合
        #         dev_label = mymodel(z)  # 获得生成的标注^dev_label
        #
        #         sup_loss = loss_fn(dev_label, label_tensor)
        #
        #         validation_loss = sup_loss
        #
        #         prediction = torch.max(F.softmax(dev_label), 1)[1]
        #         pred_label = prediction.cpu().data.numpy().squeeze()
        #         target_label = label_tensor.cpu().data.numpy()
        #         for i in range(len(annotator_id)):
        #             if pred_label[i] == target_label[i]:
        #                 count += 1
        #         num += len(annotator_id)
        # ACC = count/num
        # writer.add_scalar(tag='validation_loss', scalar_value=validation_loss.data.item(), global_step=epoch)
        # writer.add_scalar(tag='validation_acc', scalar_value=ACC, global_step=epoch)
        # print("---------------------------VALIDATION---------------------------------")
        # print(epoch, validation_loss.data.item())
        # print("validation_acc:", ACC)
    torch.save(mymodel, args.model_dir + args.mymodel_name)
