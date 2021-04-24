def sentiment_analysis(audio_path):
    import librosa
    import numpy as np
    poo = audio_path
    #step 1  feature extraction
    y, sr = librosa.load(poo, sr=44100)
    y=np.array(y)
    ar=np.array([])
    sig_mean = np.mean(abs(y))
    ar=np.append(ar,sig_mean)  # sig_mean
    ar=np.append(ar,np.std(y))
    rmse = librosa.feature.rms(y + 0.0001)[0]
    rmse=np.array(rmse)
    ar=np.append(ar,np.mean(rmse))  # rmse_mean
    ar=np.append(ar,np.std(rmse))
    silence = 0
    x=np.mean(rmse)
    l=len(rmse)
    for e in rmse:
        if e <= 0.4 * x:
            silence += 1
            silence /= float(l)
    ar=np.append(ar,silence)  # silence
    y_harmonic = librosa.effects.hpss(y)[0]
    ar=np.append(ar,(np.mean(y_harmonic) * 1000))  # harmonic (scaled by 1000)
    cl = 0.45 * sig_mean
    center_clipped = []
    for s in y:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif abs(s) < cl:
            center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    ar=np.append(ar,1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
    ar=np.append(ar,np.std(auto_corrs))
    #step 2.1 convert audio to wav format
    from pydub import AudioSegment
    sound = AudioSegment.from_file(poo)
    poowave = poo + "1"
    sound.export(poowave, format="wav")
    #step 2.2 speech to text conversion
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(poowave) as source:
        audio_text = r.listen(source)
    text = r.recognize_google(audio_text)
    #3. vectorization
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
    import pickle
    tf1 = pickle.load(open('content/features_vect.pkl', 'rb'))

    tfidf1 = TfidfVectorizer(sublinear_tf=True,min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english',vocabulary=tf1.vocabulary_)
    tex = tfidf1.fit_transform([text]).toarray()
    #step 4 loading pkl ml model files
    audio_vectors_lr = pickle.load(open('content/LR.pkl', 'rb'))
    audio_vectors_mlp = pickle.load(open('content/MLP.pkl', 'rb'))
    audio_vectors_mnb = pickle.load(open('content/MNB.pkl', 'rb'))
    audio_vectors_rf = pickle.load(open('content/RF.pkl', 'rb'))
    audio_vectors_svc = pickle.load(open('content/SVC.pkl', 'rb'))
    # #step 5 combining audio and text features and getting result array
    result = []
    arr = np.concatenate((ar,tex[0]))
    arr = arr.reshape(1,-1)
    #step 6 getting resultant array using the 5 models
    result.append(audio_vectors_lr.predict(arr))
    result.append(audio_vectors_mnb.predict(arr))
    result.append(audio_vectors_mlp.predict(arr))
    result.append(audio_vectors_rf.predict(arr))
    result.append(audio_vectors_svc.predict(arr))
    results = []
    for i in result:
        for j in i:
            results.append(j)
    # Program to find most frequent
    # element in a list

    from collections import Counter

    def most_frequent(List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]
        
    z = most_frequent(results)
    emotion = {0:"Angry",1:"Happy",2:"Sad",3:"Fear",4:"Suprise",5:"Neutral"}
    return emotion.get(z)
