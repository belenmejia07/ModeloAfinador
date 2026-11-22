import glob, numpy as np, entrenar_modelo_notas as E
rutas=sorted(glob.glob("Dataset_Notas/*.wav")); rng=np.random.default_rng(7)
por={}
for r in rutas:
    c=E.clase_de(r)
    if c: por.setdefault(c,[]).append(r)
K=5; folds=[[] for _ in range(K)]
for c,l in por.items():
    l=list(l); rng.shuffle(l)
    for i,r in enumerate(l): folds[i%K].append(r)
accs=[]; gaps=[]
for k in range(K):
    te=folds[k]; resto=[r for j in range(K) if j!=k for r in folds[j]]
    rng.shuffle(resto); nv=len(resto)//5; va,tr=resto[:nv],resto[nv:]
    Xtr,ytr=E.construir_conjunto_train(tr,rng); Xva,yva=E.construir_conjunto(va); Xte,yte=E.construir_conjunto(te)
    m=E.mlp(E.N_BINS,len(E.CLASES)); h=E.entrenar(m,Xtr,ytr,Xva,yva)
    acc_tr=float((m.predict(Xtr,verbose=0).argmax(1)==ytr).mean()); acc_te=float((m.predict(Xte,verbose=0).argmax(1)==yte).mean())
    accs.append(acc_te); gaps.append(acc_tr-acc_te); print(f"fold {k+1}: train={acc_tr:.3f} test={acc_te:.3f} epocas={len(h.history['loss'])}",flush=True)
print(f"MEDIA test={np.mean(accs):.4f} +- {np.std(accs):.4f} | brecha media train-test={np.mean(gaps):.4f}")
