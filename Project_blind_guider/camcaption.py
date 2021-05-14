import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import os
import cv2
from time import sleep
from time import time
from gtts import gTTS
import vlc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, img, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    return seq, alphas


def visualize_att(img, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = img
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


def camcap(seq,rev_word_map):
    text=[]
    for i in seq:
        text.append(rev_word_map[i])
    text=text[1:-2]    
    return ' '.join(text)

if __name__ == '__main__':

    
    model='BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map='flickr8k_preprocess/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
    beam_size=3

    url="https://192.168.42.129:8080"
    link=url+"/video"
    cap=cv2.VideoCapture('vid1.mp4')
    
    if (cap.isOpened()== False): 
    
        print("Error opening video stream or file")
        
    a=0
    o=int(time())
    
    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    f = open("call.py",'r')
    te = f.read()
    te=te.split('<Say>')
    while(1):
	   # FPS
        a+=1
        if o<int(time()):
            o=int(time())
            print('fps : ',a)
            a=0
            
        ret, frame = cap.read()       
        
        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(encoder, decoder, frame, word_map, beam_size)
        alphas = torch.FloatTensor(alphas)
        # captioning on the camera image
        text = camcap(seq,rev_word_map)
        print(text)
        frame = cv2.flip(frame,1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        s=frame.shape[0]
        
        if s > 980:
           size = 5
           scale = 2.5
           xy = (50,50)
           th = 4
        elif s > 760:
           size = 3
           scale = 2
           xy = (40,40)
           th = 3
        elif s > 520:
           size = 2
           scale = 1.5
           xy = (30,30)
           th = 2
        elif s < 500:
           size = 1
           scale = 1.3
           xy = (20,20)
           th = 2
        frame = cv2.putText(frame,text,xy, size, scale,(55,55,255),th,cv2.LINE_AA,False)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1700,920)
        cv2.imshow('frame',frame)
        #sleep(.1)
        #visualize_att(frame, seq, alphas, rev_word_map)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
        
        if key == ord('s'):
             f = open("call.py",'r')
             te = f.read()
             f.close()
             te=te.split('<Say>')  
             print(te)
             te.insert(1,'<Say>'+text)
             te=''.join(te)
             print(te)
             cf = open("cal.py",'w')
             cf.write(te)
             cf.close()
             os.system('python cal.py')
        '''
        	language= 'en'
        	output = gTTS(text=text, lang=language,slow=False)
        	output.save('output.mp3')
        	vlc_instance = vlc.Instance()
        	player = vlc_instance.media_player_new()
        	media = vlc_instance.media_new('output.mp3')
        	player.set_media(media)
        	player.play()            # very important in playing the mp3
        	print(player.get_state())# Print player's state
        	duration = player.get_length() / 1000
        	sleep(duration)
        '''
    cap.release()
    cv2.destroyAllWindows()

    #80:CE:B9:64:AC:3B
    #127.0.0.1:8080/video/mjpeg?fps=1
    #https://m2-eu5.angelcam.com/cameras/94378/streams/mjpeg/stream.mjpeg?token=eyJjYW1lcmFfaWQiOiI5NDM3OCIsInRpbWUiOjE1ODg0OTA0OTAwODc0MzQsInRpbWVvdXQiOjEyMH0%3D%2E227f273a74d7650c30649e15055d9f91aad79d240181003449b07777e503a2a2
    # Visualize caption and attention of best sequence
    #visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
