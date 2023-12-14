for task in cosine logarithm laplace
do
    for ep_adam in 200 400
    do
        for ep_lbfgs in 20 50
        do
            for lr_lbfgs in 1e-2 1e-3
            do 
                python nmlk.py --device 1 --task $task --epochs_adam $ep_adam --epochs_lbfgs $ep_lbfgs --lr_lbfgs $lr_lbfgs
            done
        done 
    done 
done 
