#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <iostream>

void data_skimming_specific() {
    /**
     * @brief Selects the TTree 'Events' from CMS Open Data file.
     */
    auto chain = std::make_unique<TChain>("Events");
    chain->Add("datasets/6357E7BC-502C-2E45-A649-73A57B651715.root");  // dataset 0
    //chain->Add("datasets/DB4AFAC8-16AD-AB48-82D2-1E9DAE8AB314.root");  // dataset 1
    //chain->Add("datasets/");  // dataset 2
    //chain->Add("datasets/");  // dataset 3
   
    /**
     * @brief Sets all branch statuses to zero.
     */
    chain->SetBranchStatus("*", 0);

    /**
     * @brief Clone full TTree structure (not the content).
     */
    TTree *newtree = chain->CloneTree(0);

    /**
     * @brief Only selects entries interesting for the ML model.
     * The entries have name and type of the original ones.
     */
    UInt_t nJet;

    Float_t MET_pt;
    Float_t MET_phi;
    Float_t MET_significance;
    Float_t GenMET_pt;

    const int maxNJets = 18;

    Float_t Jet_eta[maxNJets];
    Float_t Jet_pt[maxNJets];
    Float_t Jet_phi[maxNJets];
    Float_t Jet_mass[maxNJets];
    Float_t Jet_btagDeepFlavB[maxNJets];

    Float_t Muon_eta[maxNJets];
    Float_t Muon_pt[maxNJets];
    Float_t Muon_phi[maxNJets];
    Float_t Muon_mass[maxNJets];
    Int_t Muon_charge[maxNJets];

    /**
     * @brief Selects the previous branches, setting their
     * status to one.
     */
    chain->SetBranchStatus("nJet", 1);

    chain->SetBranchStatus("MET_pt", 1);
    chain->SetBranchStatus("MET_phi", 1);
    chain->SetBranchStatus("MET_significance", 1);
    chain->SetBranchStatus("GenMET_pt", 1);

    chain->SetBranchStatus("Jet_eta", 1);
    chain->SetBranchStatus("Jet_pt", 1);
    chain->SetBranchStatus("Jet_phi", 1);
    chain->SetBranchStatus("Jet_mass", 1);
    chain->SetBranchStatus("Jet_btagDeepFlavB", 1);

    chain->SetBranchStatus("Muon_eta", 1);
    chain->SetBranchStatus("Muon_pt", 1);
    chain->SetBranchStatus("Muon_phi", 1);
    chain->SetBranchStatus("Muon_mass", 1);
    chain->SetBranchStatus("Muon_charge", 1);
    
    /**
     * @brief Gets the address of the selected branches to copy
     * their values inside the new file.
     **/
    chain->SetBranchAddress("nJet", &nJet);

    chain->SetBranchAddress("MET_pt", &MET_pt);
    chain->SetBranchAddress("MET_phi", &MET_phi);
    chain->SetBranchAddress("MET_significance", &MET_significance);
    chain->SetBranchAddress("GenMET_pt", &GenMET_pt);

    chain->SetBranchAddress("Jet_eta", &Jet_eta);
    chain->SetBranchAddress("Jet_pt", &Jet_pt);
    chain->SetBranchAddress("Jet_phi", &Jet_phi);
    chain->SetBranchAddress("Jet_mass", &Jet_mass);
    chain->SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB);

    chain->SetBranchAddress("Muon_eta", &Muon_eta);
    chain->SetBranchAddress("Muon_pt", &Muon_pt);
    chain->SetBranchAddress("Muon_phi", &Muon_phi);
    chain->SetBranchAddress("Muon_mass", &Muon_mass);
    chain->SetBranchAddress("Muon_charge", &Muon_charge);
    
    /**
     * @param n_events Number of events in each file
     */
    Long64_t n_events = chain->GetEntries();

    /**
     * @brief Gets the maximum number of jets.
     */
    Float_t max_nJet;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);
        if (nJet > max_nJet) {
            max_nJet = nJet;
        }
    }

    std::cout << "Max number of Jets is: " << max_nJet;

    /**
     * @brief Selects only the first two jets above b-tag threshold
     * and defines them as new branches.
     */
    Float_t Jet_eta_bst, Jet_eta_bnd;
    Float_t Jet_pt_bst, Jet_pt_bnd;
    Float_t Jet_phi_bst, Jet_phi_bnd;
    Float_t Jet_mass_bst, Jet_mass_bnd;
    Float_t Jet_btag_bst, Jet_btag_bnd;

    // Best Jet
    newtree->Branch("Jet_eta_bst", &Jet_eta_bst);
    newtree->Branch("Jet_pt_bst", &Jet_pt_bst);
    newtree->Branch("Jet_phi_bst", &Jet_phi_bst);
    newtree->Branch("Jet_mass_bst", &Jet_mass_bst);
    //newtree->Branch("Jet_btag_bst", &Jet_btag_bst);

    // Second best Jet
    newtree->Branch("Jet_eta_bnd", &Jet_eta_bnd);
    newtree->Branch("Jet_pt_bnd", &Jet_pt_bnd);
    newtree->Branch("Jet_phi_bnd", &Jet_phi_bnd);
    newtree->Branch("Jet_mass_bnd", &Jet_mass_bnd);
    //newtree->Branch("Jet_btag_bnd", &Jet_btag_bnd);

    Float_t Muon_eta_st, Muon_eta_nd;
    Float_t Muon_pt_st, Muon_pt_nd;
    Float_t Muon_phi_st, Muon_phi_nd;
    Float_t Muon_mass_st, Muon_mass_nd;
    Float_t Muon_charge_st, Muon_charge_nd;

    // First Muon
    newtree->Branch("Muon_eta_st", &Muon_eta_st);
    newtree->Branch("Muon_pt_st", &Muon_pt_st);
    newtree->Branch("Muon_phi_st", &Muon_phi_st);
    newtree->Branch("Muon_mass_st", &Muon_mass_st);
    newtree->Branch("Muon_charge_st", &Muon_charge_st);
    
    // Second Muon
    newtree->Branch("Muon_eta_nd", &Muon_eta_nd);
    newtree->Branch("Muon_pt_nd", &Muon_pt_nd);
    newtree->Branch("Muon_phi_nd", &Muon_phi_nd);
    newtree->Branch("Muon_mass_nd", &Muon_mass_nd);
    newtree->Branch("Muon_charge_nd", &Muon_charge_nd);

    Float_t Muon_E_st, Muon_E_nd;
    Float_t Muon_Deltaeta;
    Float_t Muon_Deltaphi;
    Float_t Muon_DeltaR;
    Float_t Muon_InvMass;

    // Derived quantities
    newtree->Branch("Muon_Deltaeta", &Muon_Deltaeta);
    newtree->Branch("Muon_Deltaphi", &Muon_Deltaphi);
    newtree->Branch("Muon_DeltaR", &Muon_DeltaR);
    newtree->Branch("Muon_InvMass", &Muon_InvMass);

    Float_t btag_threshold = 0.3;

    for (Long64_t i = 0; i < n_events; ++i) {
        chain->GetEntry(i);

        //// First two b-tags
        //Jet_btag_bst = (nJet > 0) ? Jet_btagDeepFlavB[0] : 0.0f;
        //Jet_btag_bnd = (nJet > 1) ? Jet_btagDeepFlavB[1] : 0.0f;
//
        //if (Jet_btag_bst>btag_threshold && Jet_btag_bnd>btag_threshold) {
            // Best Jet
            Jet_eta_bst = (nJet > 0) ? Jet_eta[0] : 0.0f;
            Jet_pt_bst = (nJet > 0) ? Jet_pt[0] : 0.0f;
            Jet_phi_bst = (nJet > 0) ? Jet_phi[0] : 0.0f;
            Jet_mass_bst = (nJet > 0) ? Jet_mass[0] : 0.0f;

            // Second best Jet
            Jet_eta_bnd = (nJet > 1) ? Jet_eta[1] : 0.0f;
            Jet_pt_bnd = (nJet > 1) ? Jet_pt[1] : 0.0f;
            Jet_phi_bnd = (nJet > 1) ? Jet_phi[1] : 0.0f;
            Jet_mass_bnd = (nJet > 1) ? Jet_mass[1] : 0.0f;

            // First Muon
            Muon_eta_st = (nJet > 0) ? Muon_eta[0] : 0.0f;
            Muon_pt_st = (nJet > 0) ? Muon_pt[0] : 0.0f;
            Muon_phi_st = (nJet > 0) ? Muon_phi[0] : 0.0f;
            Muon_mass_st = (nJet > 0) ? Muon_mass[0] : 0.0f;
            Muon_charge_st = (nJet > 0) ? Muon_charge[0] : 0.0f;

            // Second Muon
            Muon_eta_nd = (nJet > 1) ? Muon_eta[1] : 0.0f;
            Muon_pt_nd = (nJet > 1) ? Muon_pt[1] : 0.0f;
            Muon_phi_nd = (nJet > 1) ? Muon_phi[1] : 0.0f;
            Muon_mass_nd = (nJet > 1) ? Muon_mass[1] : 0.0f;
            Muon_charge_nd = (nJet > 1) ? Muon_charge[1] : 0.0f;

            // Derived quantities
            Muon_Deltaeta = Muon_eta_st - Muon_eta_nd;
            Muon_Deltaphi = Muon_phi_st - Muon_phi_nd;
            Muon_DeltaR = std::sqrt(Muon_Deltaeta*Muon_Deltaeta + Muon_Deltaphi*Muon_Deltaphi);
            
            Muon_E_st = std::sqrt(Muon_pt_st*Muon_pt_st*std::cosh(Muon_eta_st)*std::cosh(Muon_eta_st)+Muon_mass_st*Muon_mass_st);
            Muon_E_nd = std::sqrt(Muon_pt_nd*Muon_pt_nd*std::cosh(Muon_eta_nd)*std::cosh(Muon_eta_nd)+Muon_mass_nd*Muon_mass_nd);
        
            Muon_InvMass = std::sqrt((Muon_E_st+Muon_E_nd)*(Muon_E_st+Muon_E_nd)-
                std::abs(Muon_pt_st+Muon_pt_nd)*std::abs(Muon_pt_st+Muon_pt_nd));
        //}
        newtree->Fill();
    }

    /**
     * @brief Creates blank new file to collect skimmed data.
     * If already existent, it recreates it.
     * 
     */
    auto skimfile = std::make_unique<TFile>("datasets/skimmed0_specific.root", "RECREATE");

    /**
     * @brief Writes the new tree than closes the new file.
     */
    newtree->Write();
    skimfile->Close();
}
