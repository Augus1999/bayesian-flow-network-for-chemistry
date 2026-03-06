### Constant

bayesianflow_for_chem.data.__VOCAB_KEYS__

&nbsp;&nbsp;&nbsp; Default SMILES and SAFE vocabulary keys.

bayesianflow_for_chem.data.__VOCAB_COUNT__

&nbsp;&nbsp;&nbsp; Default number of vocabularies of SMILES and SAFE.

bayesianflow_for_chem.data.__FASTA_VOCAB_KEYS__

&nbsp;&nbsp;&nbsp; Default FASTA sequence vocabulary keys.

bayesianflow_for_chem.data.__FASTA_VOCAB_COUNT__

&nbsp;&nbsp;&nbsp; Default number of vocabularues of FASTA.

bayesianflow_for_chem.train.__DEFAULT_MODEL_HPARAM__

&nbsp;&nbsp;&nbsp; Default hyperparameters for training a generative model.

bayesianflow_for_chem.train.__DEFAULT_GNN_HPARAM__

&nbsp;&nbsp;&nbsp; Default hyperparameters for training a GNN model for conformer searching.

bayesianflow_for_chem.train.__DEFAULT_REGRESSOR_HPARAM__

&nbsp;&nbsp;&nbsp; Default hyperparameters for training a regression or classification model.

### Tokeniser

bayesianflow_for_chem.data.__load_vocab__(_vocab_file_) &#8594; dict

&nbsp;&nbsp;&nbsp; Load vocabulary from source file.

bayesianflow_for_chem.data.__smiles2token__(_smiles_) &#8594; Tensor

&nbsp;&nbsp;&nbsp; Tokenise a SMILES and SAFE string.

bayesianflow_for_chem.data.__fasta2token__(_fasta_) &#8594; Tensor

&nbsp;&nbsp;&nbsp; Tokenise a FASTA-style sequence.

bayesianflow_for_chem.data.__split_selfies__(_selfies_) &#8594; list

&nbsp;&nbsp;&nbsp; Split a SELFIES string into individual elements.

### Dataset 

_class_ bayesianflow_for_chem.data.__CSVData__(_file_)

&nbsp;&nbsp;&nbsp; Define dataset stored in CSV file.

&nbsp;&nbsp;&nbsp; __map__(_mapping_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pass a customised mapping function to transform the data entities to tensors.

---

_class_ bayesianflow_for_chem.data.__XYZData__(_file_, _use_pbc=False_)

&nbsp;&nbsp;&nbsp; Define dataset stored in extended-XYZ file.

&nbsp;&nbsp;&nbsp; __set_smiles_keys__(_smi_keys=None_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pass a list of wanted SMILES keys to be selected in the dataset.

---

bayesianflow_for_chem.data.__collate__(_batch_) &#8594; list

&nbsp;&nbsp;&nbsp; Padding the data in one batch into the same size.

bayesianflow_for_chem.data.__graph_collate__(_batch_) &#8594; list

&nbsp;&nbsp;&nbsp; Padding the graph data in one batch into the same size.

### Model

_class_ bayesianflow_for_chem.__ChemBFN__(_num_vocab_, _channel=512_, _num_layer=12_, _num_head=8_, _dropout=0.01_)

&nbsp;&nbsp;&nbsp; Bayesian Flow Network for Chemistry model representation.

&nbsp;&nbsp;&nbsp; __enable_lora__(_r=4_, _lora_alpha=1_, _lora_dropout=0.0_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Enable LoRA parameters.

&nbsp;&nbsp;&nbsp; __reconstruction_loss__(_x_, _t_, _y_) &#8594; Tensor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute reconstruction loss.

&nbsp;&nbsp;&nbsp; __inference__(_x_, _mlp_, _embed_fn=None_) &#8594; Tensor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predict activity or property from molecular tokens.

&nbsp;&nbsp;&nbsp; <span style='color: grey'>_classmethod_</span> __from_checkpoint__(_ckpt_, _ckpt_lora=None_) &#8594; Self

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Load model weight from a checkpoint.

---

_class_ bayesianflow_for_chem.__MLP__(_size_, _class_input=False_, _dropout=0.0_)

&nbsp;&nbsp;&nbsp; __forward__(_x_) &#8594; Tensor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Do the forward pass.

&nbsp;&nbsp;&nbsp; <span style='color: grey'>_classmethod_</span> __from_checkpoint__(_ckpt_, _strict=True_) &#8594; Self

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Load model weight from a checkpoint.

---

_class_ bayesianflow_for_chem.__EnsembleChemBFN__(_base_model_path_, _lora_paths_, _cond_heads_, _adapter_weights_, _semi_autoregressive_flags_)

&nbsp;&nbsp;&nbsp; Ensemble of ChemBFN models from LoRA checkpoints.

&nbsp;&nbsp;&nbsp; __quantise__(_quantise_method=None_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Quantise the submodels.

---

_class_ bayesianflow_for_chem.geom.__EGNN__(_num_embed=120_, _channel=256_, _mol_channel=512_, _cutoff_radius=5.0_, _num_kernel=64_, _max_neighbour=15_, _num_layer=6_)

&nbsp;&nbsp;&nbsp; Equivariant Graph Neural Network representation.

&nbsp;&nbsp;&nbsp; <span style='color: grey'>_classmethod_</span> __from_checkpoint__(_ckpt_) &#8594; Self

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Load model weight from a checkpoint.

### Scorer

bayesianflow_for_chem.scorer.__smiles_valid__(_smiles_) &#8594; int

&nbsp;&nbsp;&nbsp; Return the validity of a SMILES string.

bayesianflow_for_chem.scorer.__qed_score__(_smiles_) &#8594; float

&nbsp;&nbsp;&nbsp; Return the quantitative estimate of drug-likeness score of a SMILES string.

bayesianflow_for_chem.scorer.__sa_score__(_smiles_) &#8594; float

&nbsp;&nbsp;&nbsp; Return the synthetic accessibility score of a SMILES string.

---

_class_ bayesianflow_for_chem.scorer.__Scorer__(_scorers_, _score_criteria_, _vocab_keys_, _vocab_separator=""_, _valid_checker=None_, _eta=0.001_, _name="scorer"_)

&nbsp;&nbsp;&nbsp; Scorer class that defines the scorer behaviour in the online RL.

&nbsp;&nbsp;&nbsp; __calc_score_loss__(_p_) &#8594; Tensor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Calculate the score loss.

&nbsp;&nbsp;&nbsp; <span style='color: grey'>_property_</span> __name__ &#8594; str

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Return the name of this scorer.

### Spectrum

bayesianflow_for_chem.spectra.__build_uv_vis_spectrum__(_etoscs_, _etenergies_, _lambdas_) &#8594; NDArray

&nbsp;&nbsp;&nbsp; Build UV/Vis spectrum from calculated electron transtion energies and oscillator strengths.

bayesianflow_for_chem.spectra.__build_ir_spectrum__(_vibirs_, _vibfreqs_, _nubars_) &#8594; NDArray

&nbsp;&nbsp;&nbsp; Build IR spectrum from calculated vibrational intensities and frequencies.

bayesianflow_for_chem.spectra.__build_raman_spectrum__(_vibramans_, _vibfreqs_, _nubars_) &#8594; NDArray

&nbsp;&nbsp;&nbsp; Build Raman spectrum from calculated vibrational intensities and frequencies.

bayesianflow_for_chem.spectra.__spectra_wasserstein_score__(_spectrum_u_, _spectrum_v_, _x_axis_) &#8594; NDArray

&nbsp;&nbsp;&nbsp; Return the scaled Wasserstein distance between two continuous spectra.

### Tool

bayesianflow_for_chem.tool.__test__(_model_, _mlp_, _data_, _mode_, _device=None_, _other_metrics=None_) &#8594; dict

&nbsp;&nbsp;&nbsp; Test the trained regression or classification model.

bayesianflow_for_chem.tool.__split_dataset__(_file_, _split_ratio=[8, 1, 1]_, _method="random"_) &#8594; None

&nbsp;&nbsp;&nbsp; Split a dataset stored in CSV file based on random split or scaffold split.

bayesianflow_for_chem.tool.__smaple__(_model_, _batch_size_, _sequence_size_, _sample_step=100_, _y=None_, _guidance_strength=4.0_, _device=None_, _vocab_keys=VOCAB_KEYS_, _separator=""_, _method="BFN"_, _allowed_tokens="all"_, _sort=False_) &#8594; list

&nbsp;&nbsp;&nbsp; _De novo_ generate molecules.

bayesianflow_for_chem.tool.__inpaint__(_model_, _x_, _sample_step=100_, _y=None_, _guidance_strength=4.0_, _device=None_, _vocab_keys=VOCAB_KEYS_, _separator=""_, _method="BFN"_, _allowed_tokens="all"_, _sort=False_) &#8594; list

&nbsp;&nbsp;&nbsp; Inpaint masked molecules.

bayesianflow_for_chem.tool.__optimise__(_model_, _x_, _sample_step=100_, _y=None_, _guidance_strength=4.0_, _device=None_, _vocab_keys=VOCAB_KEYS_, _separator=""_, _method="BFN"_, _allowed_tokens="all"_, _sort=False_) &#8594; list

&nbsp;&nbsp;&nbsp; Optimise template molecules.

bayesianflow_for_chem.tool.__quantise_model\___(_model_) &#8594; None

&nbsp;&nbsp;&nbsp; In-place dynamic quantise the trained model.

bayesianflow_for_chem.tool.__adjust_lora\___(_model_, _lora_scale=0.1_) &#8594; None

&nbsp;&nbsp;&nbsp; In-place adjust LoRA scaling parameter.

bayesianflow_for_chem.tool.__merge_lora\___(_model_) &#8594; None

&nbsp;&nbsp;&nbsp; In-place merge LoRA parameters into base model.

---

_class_ bayesianflow_for_chem.tool.__GeometryConverter__

&nbsp;&nbsp;&nbsp; __smiles2certesian__(_smiles_, _num_conformers_, _rdkit_ff_type="MMFF"_, _refine_with_crest=False_, _spin=0.0_) &#8594; tuple

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Guess the 3D gemoetry of the SMILES via conformer search.

&nbsp;&nbsp;&nbsp; __smiles2certesian2__(_smiles_list_, _model_, _searcher_, _search_step=100_, _lattice=None_, _device=None_) &#8594; list

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Conformer searching fully performed by ML models.

&nbsp;&nbsp;&nbsp; __cartesian2smiles__(_symbols_, _coordinates_, _charge=0_, _canonical=True_) &#8594; str

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Transform molecular geometry to SMILES string.

### Training Helper

bayesianflow_for_chem.train.__focal_loss__(_inputs_, _targets_, _alpha=None_, _gamma=2_, _reduction="mean"_) &#8594; Tensor

&nbsp;&nbsp;&nbsp; An implementation of binary and multi-class Focal Loss.

_class_ bayesianflow_for_chem.train.__Model__(_model_, _mlp=None_, _scorer=None_, _hparam=DEFAULT_MODEL_HPARAM_)

&nbsp;&nbsp;&nbsp; A `~lightning.LightningModule` wrapper of ChemBFB generative model used for training.

&nbsp;&nbsp;&nbsp; __export_model__(_workdir_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Save the trained model.

---

_class_ bayesianflow_for_chem.train.__Regressor__(_model_, _mlp_, _hparam=DEFAULT_REGRESSOR_HPARAM_)

&nbsp;&nbsp;&nbsp; A `~lightning.LightningModule` wrapper of ChemBFN regression or classification model for training.

&nbsp;&nbsp;&nbsp; __export_model__(_workdir_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Save the trained model.

---

_class_ bayesianflow_for_chem.train.__GNN__(_model_, _gnn_, _hparam=DEFAULT_GNN_HPARAM_)

&nbsp;&nbsp;&nbsp; A `~lightning.LightningModule` wrapper of conformer searching model used for training.

&nbsp;&nbsp;&nbsp; __export_model__(_workdir_) &#8594; None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Save the trained model.

