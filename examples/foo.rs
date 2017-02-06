extern crate karma;

fn main () {
    let mut hmm = karma::Karma::new(20, 10);

    hmm.randomize();

    hmm.train(&vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], None);
    hmm.train(&vec![0, 1, 4, 5, 7, 9, 0, 4, 7, 8, 2], Some(0.05));

    let out = hmm.evaluate(&vec![5, 6, 7, 8]);

    println!("1 {:?}", out);

    let testPool = vec!["1,4-Butanediol", "1P-ETH-LAD", "1P-LSD", "2-Aminoindane", "2-FA", "2-FMA", "2-Fluorodeschloroketamine", "2-Oxo-PCE", "2-methyl-2-butanol", "25B-NBOMe", "25C-NBOMe", "25I-NBOMe", "25N-NBOMe", "2C-B", "2C-B-FLY", "2C-C", "2C-D", "2C-E", "2C-H", "2C-I", "2C-P", "2C-T-2", "2C-T-7", "3-FA", "3-FEA", "3-FPM", "3-MMC", "3-MeO-PCE", "3-MeO-PCMo", "3-MeO-PCP", "4-AcO-DET", "4-AcO-DMT", "4-AcO-DiPT", "4-AcO-MET", "4-AcO-MiPT", "4-FA", "4-FMA", "4-HO-DET", "4-HO-DPT", "4-HO-DiPT", "4-HO-EPT", "4-HO-MET", "4-HO-MPT", "4-HO-MiPT", "4-MeO-PCP", "4F-EPH", "5-APB", "5-Hydroxytryptophan", "5-MAPB", "5-MeO-DALT", "5-MeO-DMT", "5-MeO-DiBF", "5-MeO-DiPT", "5-MeO-MiPT", "5F-AKB48", "5F-PB-22", "6-APB", "6-APDB", "A-PHP", "A-PVP", "AB-FUBINACA", "AL-LAD", "ALD-52", "Acetylfentanyl", "Alcohol", "Alpha-GPC", "Alprazolam", "Amphetamine", "Armodafinil", "Ayahuasca", "Benzydamine", "Bufotenin", "Buprenorphine", "Butylone", "Caffeine", "Cannabis", "Carisoprodol", "Choline bitartrate", "Citicoline", "Clonazepam", "Clonazolam", "Cocaine", "Codeine", "Coluracetam", "Creatine", "DET", "DMT", "DOB", "DOC", "DOI", "DOM", "DPT", "DXM & DPH in combination", "Dehydroxyfluorafinil", "Deschloroetizolam", "Deschloroketamine", "Desoxypipradol", "Dextromethorphan", "Dextropropoxyphene", "DiPT", "Diazepam", "Diclazepam", "Dihydrocodeine", "Diphenhydramine", "Diphenidine", "ETH-LAD", "Efavirenz", "Ephenidine", "Escaline", "Ethyl-Hexedrone", "Ethylcathinone", "Ethylone", "Ethylphenidate", "Etizolam", "F-Phenibut", "Fentanyl", "Flubromazepam", "Flubromazolam", "GBL", "GHB", "Gabapentin", "Haloperidol", "Heroin", "Hydrocodone", "Hydromorphone", "Ibogaine", "Isopropylphenidate", "JWH-073", "Ketamine", "Kratom", "LSA", "LSD", "LSZ", "Lisdexamfetamine", "Lorazepam", "MCPP", "MDA", "MDAI", "MDEA", "MDMA", "MDPV", "MET", "MPT", "Melatonin", "Mephedrone", "Methadone", "Methallylescaline", "Methamphetamine", "Methaqualone", "Methiopropamine", "Methoxetamine", "Methoxphenidine", "Methylnaphthidate", "Methylone", "Methylphenidate", "Metizolam", "Mexedrone", "MiPT", "Mirtazapine", "Modafinil", "N-Acetylcysteine", "NM-2-AI", "Naloxone", "Nicotine", "Nifoxipam", "Nitrous Oxide", "Noopept", "O-Desmethyltramadol", "Oxiracetam", "Oxycodone", "Oxymorphone", "PCP", "PMMA", "PRO-LAD", "Pentobarbital", "Pethidine", "Phenibut", "Phenobarbital", "Piracetam", "Pramiracetam", "Pregabalin", "Prolintane", "Propylhexedrine", "Proscaline", "Psilocin", "Pyrazolam", "Quetiapine", "RTI-111", "Risperidone", "STS-135", "Salvinorin A", "Secobarbital", "Sufentanil", "THJ-018", "THJ-2201", "TMA-2", "TMA-6", "Tapentadol", "Temazepam", "Theanine", "Tianeptine", "Tramadol", "Tyrosine", "U-47700", "Zolpidem", "Zopiclone"];

    let mut hmm = karma::Karma::new((testPool.len() as i64) * 4, testPool.len() as i64);

    hmm.randomize();

    let r_vec = vec![
        vec![10, 71, 37, 42, 88, 89, 195, 106, 10, 99, 43],
        vec![15, 71, 88, 39, 40, 85, 61, 32, 1, 50, 32],
        vec![130, 62, 37, 49, 16, 91, 106, 16, 61, 15, 69],
        vec![71, 196, 130, 51, 195, 2, 91, 22, 61, 20, 51],
        vec![196, 71, 21, 146, 157, 39, 204, 88, 37, 43, 69],
        vec![69, 196, 91, 136, 91, 12, 195, 42, 39, 30, 49],
        vec![204, 108, 51, 18, 183, 141, 16, 52, 184, 204, 14],
        vec![112, 4, 35, 96, 152, 179, 3, 23, 161, 81, 140],
        vec![179, 59, 59, 74, 74, 109, 23, 133, 96, 147, 23],
        vec![5, 35, 81, 187, 140, 112, 163, 59, 156, 35, 58],
        vec![153, 202, 147, 109, 111, 133, 96, 25, 96, 156, 23],
        vec![161, 23, 147, 111, 149, 163, 59, 36, 73, 147, 168],
        vec![163, 144, 181, 182, 181, 4, 3, 202, 81, 152, 23],
        vec![163, 4, 126, 58, 156, 4, 149, 81, 149, 133, 163],
        vec![191, 8, 205, 94, 113, 117, 76, 64, 94, 119, 76],
        vec![164, 0, 116, 118, 134, 185, 101, 113, 118, 148, 80],
        vec![94, 113, 164, 113, 66, 8, 191, 176, 164, 117, 64],
        vec![100, 185, 64, 174, 101, 101, 134, 198, 79, 185, 113],
        vec![66, 205, 198, 174, 94, 119, 76, 76, 0, 101, 148],
        vec![80, 100, 94, 94, 114, 177, 64, 134, 0, 79, 64],
        vec![174, 164, 164, 116, 114, 205, 148, 66, 79, 117, 176]
    ];

    for t in r_vec.iter() {
        hmm.train(t, Some(0.05));
    }

    let t_vec = vec![
        vec![76, 0, 101, 148],
        vec![111, 149, 163, 59],
        vec![16, 91, 106, 16],
        vec![109, 111, 133, 96],
        vec![164, 116, 114, 205, 94],
        vec![164, 116, 114, 205, 198],
        vec![164, 116, 114, 205, 148],
    ];

    for t in t_vec.iter() {
        let out = hmm.evaluate(&t);

        println!("2 {:?}", out);
    }
}