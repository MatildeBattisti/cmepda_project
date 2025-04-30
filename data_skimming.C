#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void data_skimming() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("datasets/6357E7BC-502C-2E45-A649-73A57B651715.root");
   
    /**
     * @brief Sets all branch statuses to zero.
     */
    chain->SetBranchStatus("*", 0);

    /**
     * @brief Only selects entries interesting for the ML model.
     * The entries have name and type of the original ones.
     */
    UInt_t nJet;
    Float_t MET_pt;
    Float_t MET_phi;
    Float_t MET_covXX;
    Float_t MET_covXY;
    Float_t MET_covYY;
    Float_t MET_significance;

    const int maxNJets = 18;

    Float_t Jet_eta[maxNJets];
    Float_t Jet_pt[maxNJets];
    Float_t Jet_phi[maxNJets];
    Float_t Jet_mass[maxNJets];

    /*
     * @brief Selects the previous branches, setting their
     * status to one.
     */
    chain->SetBranchStatus("nJet", 1);
    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_mass", 1);
    chain->SetBranchStatus("MET_pt", 1);
    chain->SetBranchStatus("MET_phi", 1);
    chain->SetBranchStatus("MET_covXX", 1);
    chain->SetBranchStatus("MET_covXY", 1);
    chain->SetBranchStatus("MET_covYY", 1);
    chain->SetBranchStatus("MET_significance", 1);
  
    /**
     * @brief Gets the address of the selected branches to copy
     * their values inside the new file.
     **/
    chain->SetBranchAddress("nJet", &nJet);
    chain->SetBranchAddress("Jet_eta", &Jet_eta);
    chain->SetBranchAddress("Jet_pt", &Jet_pt);
    chain->SetBranchAddress("Jet_phi", &Jet_phi);
    chain->SetBranchAddress("Jet_mass", &Jet_mass);
    chain->SetBranchAddress("MET_pt", &MET_pt);
    chain->SetBranchAddress("MET_phi", &MET_phi);
    chain->SetBranchAddress("MET_covXX", &MET_covXX);
    chain->SetBranchAddress("MET_covXY", &MET_covXY);
    chain->SetBranchAddress("MET_covYY", &MET_covYY);
    chain->SetBranchAddress("MET_significance", &MET_significance);

    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);

    /**
     * @brief Selects only the first three jets and
     * defines them as new branches.
     * Creates log array for each entry.
     * Log array for MET_pt entry.
     */
    Float_t Jet_eta_bst, Jet_eta_bnd, Jet_eta_brd;
    Float_t Jet_pt_bst, Jet_pt_bnd, Jet_pt_brd;
    Float_t Jet_phi_bst, Jet_phi_bnd, Jet_phi_brd;
    Float_t Jet_mass_bst, Jet_mass_bnd, Jet_mass_brd;

    Float_t Jet_eta_bst_log, Jet_eta_bnd_log, Jet_eta_brd_log;
    Float_t Jet_pt_bst_log, Jet_pt_bnd_log, Jet_pt_brd_log;
    Float_t Jet_phi_bst_log, Jet_phi_bnd_log, Jet_phi_brd_log;
    Float_t Jet_mass_bst_log, Jet_mass_bnd_log, Jet_mass_brd_log;

    Float_t MET_pt_log;

    Float_t m_hh;

    // Best Jet
    newtree->Branch("Jet_eta_bst", &Jet_eta_bst);
    newtree->Branch("Jet_pt_bst", &Jet_pt_bst);
    newtree->Branch("Jet_phi_bst", &Jet_phi_bst);
    newtree->Branch("Jet_mass_bst", &Jet_mass_bst);

    newtree->Branch("Jet_eta_bst_log", &Jet_eta_bst_log);
    newtree->Branch("Jet_pt_bst_log", &Jet_pt_bst_log);
    newtree->Branch("Jet_phi_bst_log", &Jet_phi_bst_log);
    newtree->Branch("Jet_mass_bst_log", &Jet_mass_bst_log);

    // Second best Jet
    newtree->Branch("Jet_eta_bnd", &Jet_eta_bnd);
    newtree->Branch("Jet_pt_bnd", &Jet_pt_bnd);
    newtree->Branch("Jet_phi_bnd", &Jet_phi_bnd);
    newtree->Branch("Jet_mass_bnd", &Jet_mass_bnd);

    newtree->Branch("Jet_eta_bnd_log", &Jet_eta_bnd_log);
    newtree->Branch("Jet_pt_bnd_log", &Jet_pt_bnd_log);
    newtree->Branch("Jet_phi_bnd_log", &Jet_phi_bnd_log);
    newtree->Branch("Jet_mass_bnd_log", &Jet_mass_bnd_log);

    // Third best Jet
    newtree->Branch("Jet_eta_brd", &Jet_eta_brd);
    newtree->Branch("Jet_pt_brd", &Jet_pt_brd);
    newtree->Branch("Jet_phi_brd", &Jet_phi_brd);
    newtree->Branch("Jet_mass_brd", &Jet_mass_brd);

    newtree->Branch("Jet_eta_brd_log", &Jet_eta_brd_log);
    newtree->Branch("Jet_pt_brd_log", &Jet_pt_brd_log);
    newtree->Branch("Jet_phi_brd_log", &Jet_phi_brd_log);
    newtree->Branch("Jet_mass_brd_log", &Jet_mass_brd_log);

    // MET_pt
    newtree->Branch("MET_pt_log", &MET_pt_log);

    // m_hh
    newtree->Branch("m_hh", &m_hh);

    Float_t min_MET_pt;
    Float_t max_nJet;

    Long64_t n_events = chain->GetEntries();

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);

        // Max number of Jets
        if (nJet > max_nJet) {
            max_nJet = nJet;
        }

        Float_t eps = 6;

        // Best Jet
        Jet_eta_bst = (nJet > 0) ? Jet_eta[0] : 0.0f;
        Jet_pt_bst = (nJet > 0) ? Jet_pt[0] : 0.0f;
        Jet_phi_bst = (nJet > 0) ? Jet_phi[0] : 0.0f;
        Jet_mass_bst = (nJet > 0) ? Jet_mass[0] : 0.0f;

        Jet_eta_bst_log = std::log(Jet_eta_bst + eps);
        Jet_pt_bst_log = std::log(Jet_pt_bst + eps);
        Jet_phi_bst_log = std::log(Jet_phi_bst + eps);
        Jet_mass_bst_log = std::log(Jet_mass_bst + eps);
        
        // Second best Jet
        Jet_eta_bnd = (nJet > 1) ? Jet_eta[1] : 0.0f;
        Jet_pt_bnd = (nJet > 1) ? Jet_pt[1] : 0.0f;
        Jet_phi_bnd = (nJet > 1) ? Jet_phi[1] : 0.0f;
        Jet_mass_bnd = (nJet > 1) ? Jet_mass[1] : 0.0f;

        Jet_eta_bnd_log = std::log(Jet_eta_bnd + eps);
        Jet_pt_bnd_log = std::log(Jet_pt_bnd + eps);
        Jet_phi_bnd_log = std::log(Jet_phi_bnd + eps);
        Jet_mass_bnd_log = std::log(Jet_mass_bnd + eps);
        
        // Third best Jet
        Jet_eta_brd = (nJet > 2) ? Jet_eta[2] : 0.0f;
        Jet_pt_brd = (nJet > 2) ? Jet_pt[2] : 0.0f;
        Jet_phi_brd = (nJet > 2) ? Jet_phi[2] : 0.0f;
        Jet_mass_brd = (nJet > 2) ? Jet_mass[2] : 0.0f;

        Jet_eta_brd_log = std::log(Jet_eta_brd + eps);
        Jet_pt_brd_log = std::log(Jet_pt_brd + eps);
        Jet_phi_brd_log = std::log(Jet_phi_brd + eps);
        Jet_mass_brd_log = std::log(Jet_mass_brd + eps);

        // MET_pt
        if (MET_pt < min_MET_pt) {
            min_MET_pt = MET_pt;
        }
        MET_pt_log = std::log(MET_pt + min_MET_pt + 1);
        
        // m_hh
        m_hh = Jet_mass_bst + Jet_mass_bnd;

        // Fill Tree with new entries
        newtree->Fill();
    }

    std::cout << "Max number of Jets is:" << max_nJet ;

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("datasets/skimmed2.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}