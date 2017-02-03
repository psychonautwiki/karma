const HMM = ((() => {
    const ARR = (typeof Float64Array) == 'function' ? Float64Array : Array;

    const _choose = function _choose(a, ao, as, al){
        let i;
        let x = Math.random();
        for(i=0; i<al; i++){
            x -= a[ao+i*as];
            if(x < 0) return i;
        }
        return al-1;
    };

    class HMM {
        constructor(N, O) {
            this.N = N;this.O = O;

            this.init_probs = new ARR(N);
            this.next_probs = new ARR(N*N); // curr*N + next
            this.out_probs = new ARR(N*O); // out*N + curr

            let obj;
            let i;
            let j;
            for(i=0; i<N*N; i++) this.next_probs[i] = 1/N;
            for(i=0; i<N*O; i++) this.out_probs[i] = 1/O;
            for(i=0; i<N; i++) this.init_probs[i] = 1/N;

            // for computing
            this._resize_tmp(16);
        }

        _resize_tmp(s) {
            if(s <= this._max_s) return;
            this._max_s = s;
            this._tmp = new ARR(this.N*(3*this._max_s+this.N));
        }

        randomize() {
            let i;
            let j;
            let k;
            let x;
            const N = this.N;
            for(k=0; k<3*N; k++){
                i = 0|Math.random()*N;
                j = 0|Math.random()*N;
                if(i==j) continue;
                if(this.init_probs[i] + this.init_probs[j] >= 1) continue;
                x = Math.random()*this.init_probs[i];
                this.init_probs[i] -= x;
                this.init_probs[j] += x;
            }
        }

        simulate(in_state) {
            const N = this.N;
            const O = this.O;
            if(in_state == null) in_state = -1;
            const next_state = in_state == -1 ? _choose(this.init_probs, 0, 1, N) : _choose(this.next_probs, in_state*N, 1, N);
            const next_output = _choose(this.out_probs, next_state, N, O);
            return [next_state, next_output];
        }

        generate(process) {
            let curr_state = -1;
            const result_str = [];
            for(let i=0;;i++){
                const v = this.simulate(curr_state);
                curr_state = v[0];
                result_str.push(v[1]);
                if(!process(v[1], i)) return result_str;
            }
        }

        evaluate(outputs) {
            if(outputs.length == 0) return 1;

            const N = this.N;
            const init_probs = this.init_probs;
            const next_probs = this.next_probs;
            const out_probs = this.out_probs;

            const alphas = this._tmp;

            let t;
            let i;
            let j;
            let k;
            let l;
            let sum;
            let output;

            for(i=0,j=outputs[0]*N; i<N; i++,j++) alphas[i] = init_probs[i] * out_probs[j];

            for(t=1; t<outputs.length; t++){
                output = outputs[t];
                for(j=0,k=output*N; j<N; j++,k++){
                    for(sum=i=0,l=j; i<N; i++, l+=N) sum += alphas[i] * next_probs[l];
                    alphas[N+j] = sum * out_probs[k];
                }
                if(++t >= outputs.length) break;

                output = outputs[t];
                for(j=0,k=output*N; j<N; j++,k++){
                    for(sum=i=0,l=j; i<N; i++, l+=N) sum += alphas[N+i] * next_probs[l];
                    alphas[j] = sum * out_probs[k];
                }
            }

            if(t&1) for(sum=i=0; i<N; i++) sum += alphas[i];
            else for(sum=i=0; i<N; i++) sum += alphas[N+i];
            return sum;
        }

        train(outputs, rate) {
            if(rate == null) rate = .05;

            let i;
            let j;
            let k;
            let l;
            let sum;
            const init = this.init_probs;
            const next_probs = this.next_probs;
            const out_probs = this.out_probs;

            const n_next_probs = this._n_next_probs;

            const N = this.N;
            const O = this.O;
            const S = outputs.length;
            const outputs_N = outputs.map(v => v*N);

            this._resize_tmp(S);
            const BETA = N*S;
            const GAMMA = 2*BETA;
            const KAPPA = 3*BETA;
            const arr = this._tmp;

            // Expectation - step 1 (computing alpha and beta)
            for(j=0; j<N; j++) arr[j] = init[j]*out_probs[j+outputs_N[0]];
            for(i=1; i<S; i++){
                for(j=l=0; j<N; j++){
                    for(k=sum=0; k<N; k++,l++){
                        arr[KAPPA+l] = 0; // init kappa here
                        sum += arr[(i-1)*N+k]*next_probs[k*N+j];
                    }
                    arr[i*N+j] = sum*out_probs[j+outputs_N[i]];
                }
            }
            for(j=0,l=(S-1)*N; j<N; j++,l++){
                arr[BETA+l] = 1;
            }
            for(i=S-1; i-->0; ){
                for(j=0,l=i*N; j<N; j++, l++){
                    arr[BETA+l] = 0;
                    for(k=0; k<N; k++)
                        arr[BETA+l] += next_probs[j*N+k]*out_probs[k+outputs_N[i+1]]*arr[BETA+(i+1)*N+k];
                }
            }

            // Expectation - step 2 (computing gamma and kappa)
            for(i=0; i<S; i++){
                for(k=sum=0; k<N; k++) sum += arr[i*N+k]*arr[BETA+i*N+k];
                for(j=0; j<N; j++){
                    arr[GAMMA+i*N+j] = arr[i*N+j]*arr[BETA+i*N+j]/sum;
                }
                if(i==S-1) break;
                for(j=sum=0; j<N; j++) for(k=0; k<N; k++){
                    sum += arr[i*N+j]*next_probs[j*N+k]*out_probs[k+outputs_N[i+1]]*arr[BETA+(i+1)*N+k];
                }
                for(l=j=0; j<N; j++) for(k=0; k<N; k++,l++){
                    arr[KAPPA+l] += arr[i*N+j]*next_probs[l]*out_probs[k+outputs_N[i+1]]*arr[BETA+(i+1)*N+k]/sum;
                }
            }

            // Maximum likelihood
            let x;

            const p=[];
            let del;
            for(l=i=0; i<N; i++){
                for(k=sum=0; k<S-1; k++) sum += arr[GAMMA+k*N+i];
                for(j=0; j<N; j++,l++){
                    del = arr[KAPPA+l]/sum-next_probs[l];
                    next_probs[l] += del*rate;
                }
                sum += arr[GAMMA+(S-1)*N+i];
                for(j=0; j<O; j++){
                    for(k=x=0; k<S; k++) if(outputs[k]==j) x += arr[GAMMA+k*N+i];
                    del = x/sum-out_probs[i+j*N];
                    out_probs[i+j*N] += del*rate;
                }
                del = arr[GAMMA+i]-init[i];
                init[i] += del*rate;
            }
        }

        static parse(str) {
            const json = JSON.parse(str);
            const hmm = new HMM(json[0], json[1]);
            hmm.next_probs = ARR.call(null, json[2]);
            hmm.out_probs = ARR.call(null, json[3]);
            hmm.init_probs = ARR.call(null, json[4]);

            return hmm;
        }
    }

    HMM.prototype.init_probs = null;
    HMM.prototype.next_probs = null;
    HMM.prototype.out_probs = null;
    HMM.prototype._tmp = null;
    HMM.prototype._max_s = 0;

    return HMM;
}))();

if(typeof module !== 'undefined' && typeof require !== 'undefined'){
    module.exports = HMM;
}