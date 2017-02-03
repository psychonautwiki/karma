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

const testPool = ["1,4-Butanediol","1P-ETH-LAD","1P-LSD","2-Aminoindane","2-FA","2-FMA","2-Fluorodeschloroketamine","2-Oxo-PCE","2-methyl-2-butanol","25B-NBOMe","25C-NBOMe","25I-NBOMe","25N-NBOMe","2C-B","2C-B-FLY","2C-C","2C-D","2C-E","2C-H","2C-I","2C-P","2C-T-2","2C-T-7","3-FA","3-FEA","3-FPM","3-MMC","3-MeO-PCE","3-MeO-PCMo","3-MeO-PCP","4-AcO-DET","4-AcO-DMT","4-AcO-DiPT","4-AcO-MET","4-AcO-MiPT","4-FA","4-FMA","4-HO-DET","4-HO-DPT","4-HO-DiPT","4-HO-EPT","4-HO-MET","4-HO-MPT","4-HO-MiPT","4-MeO-PCP","4F-EPH","5-APB","5-Hydroxytryptophan","5-MAPB","5-MeO-DALT","5-MeO-DMT","5-MeO-DiBF","5-MeO-DiPT","5-MeO-MiPT","5F-AKB48","5F-PB-22","6-APB","6-APDB","A-PHP","A-PVP","AB-FUBINACA","AL-LAD","ALD-52","Acetylfentanyl","Alcohol","Alpha-GPC","Alprazolam","Amphetamine","Armodafinil","Ayahuasca","Benzydamine","Bufotenin","Buprenorphine","Butylone","Caffeine","Cannabis","Carisoprodol","Choline bitartrate","Citicoline","Clonazepam","Clonazolam","Cocaine","Codeine","Coluracetam","Creatine","DET","DMT","DOB","DOC","DOI","DOM","DPT","DXM & DPH in combination","Dehydroxyfluorafinil","Deschloroetizolam","Deschloroketamine","Desoxypipradol","Dextromethorphan","Dextropropoxyphene","DiPT","Diazepam","Diclazepam","Dihydrocodeine","Diphenhydramine","Diphenidine","ETH-LAD","Efavirenz","Ephenidine","Escaline","Ethyl-Hexedrone","Ethylcathinone","Ethylone","Ethylphenidate","Etizolam","F-Phenibut","Fentanyl","Flubromazepam","Flubromazolam","GBL","GHB","Gabapentin","Haloperidol","Heroin","Hydrocodone","Hydromorphone","Ibogaine","Isopropylphenidate","JWH-073","Ketamine","Kratom","LSA","LSD","LSZ","Lisdexamfetamine","Lorazepam","MCPP","MDA","MDAI","MDEA","MDMA","MDPV","MET","MPT","Melatonin","Mephedrone","Methadone","Methallylescaline","Methamphetamine","Methaqualone","Methiopropamine","Methoxetamine","Methoxphenidine","Methylnaphthidate","Methylone","Methylphenidate","Metizolam","Mexedrone","MiPT","Mirtazapine","Modafinil","N-Acetylcysteine","NM-2-AI","Naloxone","Nicotine","Nifoxipam","Nitrous Oxide","Noopept","O-Desmethyltramadol","Oxiracetam","Oxycodone","Oxymorphone","PCP","PMMA","PRO-LAD","Pentobarbital","Pethidine","Phenibut","Phenobarbital","Piracetam","Pramiracetam","Pregabalin","Prolintane","Propylhexedrine","Proscaline","Psilocin","Pyrazolam","Quetiapine","RTI-111","Risperidone","STS-135","Salvinorin A","Secobarbital","Sufentanil","THJ-018","THJ-2201","TMA-2","TMA-6","Tapentadol","Temazepam","Theanine","Tianeptine","Tramadol","Tyrosine","U-47700","Zolpidem","Zopiclone"];

const mapToTestPool = vec => vec.map(substance => testPool.indexOf(substance));

{
    // Create a HMM with 20 states and 10 characters.
    var hmm = new HMM(testPool.length * 4, testPool.length);

    // Randomize HMM (randomizing initial probabilities of each states)
    //hmm.randomize();

    const testVectors = [
        // psychedelics

        ["25C-NBOMe", "Bufotenin", "4-HO-DET", "4-HO-MPT", "DOC", "DOI", "TMA-2", "Efavirenz", "25C-NBOMe", "DiPT", "4-HO-MiPT"],
        ["2C-C", "Bufotenin", "DOC", "4-HO-DiPT", "4-HO-EPT", "DET", "AL-LAD", "4-AcO-DiPT", "1P-ETH-LAD", "5-MeO-DMT", "4-AcO-DiPT"],
        ["LSA", "ALD-52", "4-HO-DET", "5-MeO-DALT", "2C-D", "DPT", "Efavirenz", "2C-D", "AL-LAD", "2C-C", "Ayahuasca"],
        ["Bufotenin", "TMA-6", "LSA", "5-MeO-DiBF", "TMA-2", "1P-LSD", "DPT", "2C-T-7", "AL-LAD", "2C-P", "5-MeO-DiBF"],
        ["TMA-6", "Bufotenin", "2C-T-2", "Methallylescaline", "MiPT", "4-HO-DiPT", "Zolpidem", "DOC", "4-HO-DET", "4-HO-MiPT", "Ayahuasca"],
        ["Ayahuasca", "TMA-6", "DPT", "MDA", "DPT", "25N-NBOMe", "TMA-2", "4-HO-MPT", "4-HO-DiPT", "4-AcO-DET", "5-MeO-DALT"],
        ["Zolpidem", "Escaline", "5-MeO-DiBF", "2C-H", "Proscaline", "MET", "2C-D", "5-MeO-DiPT", "Psilocin", "Zolpidem", "2C-B-FLY"],

        // stimulants

        ["Ethylphenidate", "2-FA", "4-FA", "Desoxypipradol", "Methylnaphthidate", "Pramiracetam", "2-Aminoindane", "3-FA", "NM-2-AI", "Cocaine", "MDPV"],
        ["Pramiracetam", "A-PVP", "A-PVP", "Caffeine", "Caffeine", "Ethyl-Hexedrone", "3-FA", "Lisdexamfetamine", "Desoxypipradol", "Methamphetamine", "3-FA"],
        ["2-FMA", "4-FA", "Cocaine", "RTI-111", "MDPV", "Ethylphenidate", "Nicotine", "A-PVP", "Mexedrone", "4-FA", "A-PHP"],
        ["Methylone", "Tyrosine", "Methamphetamine", "Ethyl-Hexedrone", "Ethylone", "Lisdexamfetamine", "Desoxypipradol", "3-FPM", "Desoxypipradol", "Mexedrone", "3-FA"],
        ["NM-2-AI", "3-FA", "Methamphetamine", "Ethylone", "Methiopropamine", "Nicotine", "A-PVP", "4-FMA", "Butylone", "Methamphetamine", "Oxiracetam"],
        ["Nicotine", "Mephedrone", "Prolintane", "Propylhexedrine", "Prolintane", "2-FA", "2-Aminoindane", "Tyrosine", "Cocaine", "Methylnaphthidate", "3-FA"],
        ["Nicotine", "2-FA", "Isopropylphenidate", "A-PHP", "Mexedrone", "2-FA", "Methiopropamine", "Cocaine", "Methiopropamine", "Lisdexamfetamine", "Nicotine"],

        // depressants

        ["Secobarbital", "2-methyl-2-butanol", "Zopiclone", "Deschloroetizolam", "Etizolam", "Flubromazolam", "Carisoprodol", "Alcohol", "Deschloroetizolam", "GHB", "Carisoprodol"],
        ["Nifoxipam", "1,4-Butanediol", "Flubromazepam", "GBL", "Lorazepam", "Pyrazolam", "Diclazepam", "Etizolam", "GBL", "Methaqualone", "Clonazolam"],
        ["Deschloroetizolam", "Etizolam", "Nifoxipam", "Etizolam", "Alprazolam", "2-methyl-2-butanol", "Secobarbital", "Phenibut", "Nifoxipam", "Flubromazolam", "Alcohol"],
        ["Diazepam", "Pyrazolam", "Alcohol", "Pentobarbital", "Diclazepam", "Diclazepam", "Lorazepam", "Temazepam", "Clonazepam", "Pyrazolam", "Etizolam"],
        ["Alprazolam", "Zopiclone", "Temazepam", "Pentobarbital", "Deschloroetizolam", "GHB", "Carisoprodol", "Carisoprodol", "1,4-Butanediol", "Diclazepam", "Methaqualone"],
        ["Clonazolam", "Diazepam", "Deschloroetizolam", "Deschloroetizolam", "F-Phenibut", "Phenobarbital", "Alcohol", "Lorazepam", "1,4-Butanediol", "Clonazepam", "Alcohol"],
        ["Pentobarbital", "Nifoxipam", "Nifoxipam", "Flubromazepam", "F-Phenibut", "Zopiclone", "Methaqualone", "Alprazolam", "Clonazepam", "Flubromazolam", "Phenibut"]
    ].map(vec => hmm.train(mapToTestPool(vec), 0.05))

    console.log(
        hmm.evaluate(mapToTestPool(["Carisoprodol", "1,4-Butanediol", "Diclazepam", "Methaqualone"])),
        hmm.evaluate(mapToTestPool(["Ethylone", "Methiopropamine", "Nicotine", "A-PVP"])),
        hmm.evaluate(mapToTestPool(["2C-D", "DPT", "Efavirenz", "2C-D"]))
    );
}