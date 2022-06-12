import torch
batch_count = 60
torch.cuda.empty_cache()


    for batch in tqdm(range(batch_count)):
        hr_imagesList = [
            img for img in HR_images_list[batch*batch_size:(batch+1)*batch_size]]
        lr_images = loadLRImages(hr_imagesList, hr_path)/255
        hr_images = loadImages(hr_imagesList, hr_path, True)/255

        disc.zero_grad()

        gen_out = gen(torch.from_numpy(lr_images).to(cuda).float())
        _, f_label = disc(gen_out)
        _, r_label = disc(torch.from_numpy(hr_images).to(cuda).float())
        d1_loss = (disc_loss(f_label, torch.zeros_like(
            f_label, dtype=torch.float)))
        d2_loss = (disc_loss(r_label, torch.ones_like(
            r_label, dtype=torch.float)))
        # d_loss = d1_loss+d2_loss
        d2_loss.backward()
        d1_loss.backward(retain_graph=True)
        # print(d1_loss,d2_loss)
#         d_loss.backward(retain_graph=True)
        disc_optimizer.step()

        gen.zero_grad()
        g_loss = gen_loss(f_label.data, torch.ones_like(
            f_label, dtype=torch.float))
        v_loss = vgg_loss(vgg.features[:7](gen_out), vgg.features[:7](
            torch.from_numpy(hr_images).to(cuda).float()))
        m_loss = mse_loss(gen_out, torch.from_numpy(
            hr_images).to(cuda).float())

        generator_loss = g_loss + v_loss + m_loss
        # v_loss.backward(retain_graph=True)
        # m_loss.backward(retain_graph=True)
        # g_loss.backward()
        # print(generator_loss)

        generator_loss.backward()
        gen_optimizer.step()

        d1loss_list.append(d1_loss.item())
        d2loss_list.append(d2_loss.item())

        gloss_list.append(g_loss.item())
        vloss_list.append(v_loss.item())
        mloss_list.append(m_loss.item())


#         print("d1Loss ::: "+str((d1_loss.item()))+" d2Loss ::: "+str((d2_loss.item())))
#         print("gloss ::: "+str((g_loss.item()))+" vloss ::: "+str((v_loss.item()))+" mloss ::: "+str((m_loss.item())))
    print("Epoch ::::  "+str(epoch+1)+"  d1_loss ::: " +
          str(np.mean(d1loss_list))+"  d2_loss :::"+str(np.mean(d2loss_list)))
    print("genLoss ::: "+str(np.mean(gloss_list))+"  vggLoss ::: " +
          str(np.mean(vloss_list))+"  MeanLoss  ::: "+str(np.mean(mloss_list)))

    if(epoch % 3 == 0):

        checkpoint = {'model': Generator(),
                      'input_size': 64,
                      'output_size': 256,
                      'state_dict': gen.state_dict()}
        torch.save(checkpoint, os.path.join(
            weight_file, "SR"+str(epoch+1)+".pth"))
        torch.cuda.empty_cache()

        out_images = imagePostProcess(
            IMAGES[-2:], os.path.join(weight_file, "SR"+str(epoch+1)+".pth"))
#         print(out_images.shape)
#         test_images = loadLRImages(images[:-3],hr_path)/255
#         test_images = np.reshape(test_images,(test_images[0],test_images.shape[3],test_images.shape[1],test_images.shape[2]))
#         out_images = gen(torch.from_numpy(test_images).to(cuda).float())
#         out_images = np.reshape(out_images,(out_images[0],out_images[2],out_images[3],out_images[1]))
        show_samples(out_images)
