"""
Materials science module for SCICanvas.

This module provides state-of-the-art neural network models and predictors
for materials science applications including:
- Crystal structure prediction
- Materials property prediction
- Catalyst design and optimization
- Phase diagram analysis
- Electronic structure prediction
"""

from .models import (
    CrystalGraphConvNet,
    MaterialsTransformer,
    CatalystDesignNet,
    PhasePredictor,
    ElectronicStructureNet
)

from .predictors import (
    PropertyPredictor,
    StructurePredictor,
    CatalystDesigner,
    PhaseAnalyzer,
    ElectronicStructureAnalyzer
)

from .data import (
    MaterialsDataset,
    MaterialsDataLoader,
    load_example_data,
    generate_synthetic_materials_data,
    load_materials_project_sample,
    load_oqmd_sample,
    create_crystal_graph,
    save_materials_data,
    load_materials_data,
    compute_material_fingerprint,
    create_materials_database
)

from .preprocessing import (
    normalize_structure,
    compute_structural_features,
    compute_coordination_numbers,
    compute_elemental_features,
    augment_structure,
    filter_structures,
    standardize_features,
    compute_radial_distribution_function,
    compute_bond_angles,
    detect_defects
)

from .visualization import (
    plot_crystal_structure,
    plot_property_distribution,
    plot_property_correlation,
    plot_phase_diagram,
    plot_band_structure,
    plot_catalyst_performance,
    plot_structure_features,
    plot_radial_distribution_function,
    plot_formation_energy_vs_composition,
    plot_prediction_results,
    plot_materials_database_overview,
    create_materials_report,
    plot_defect_analysis
)

__all__ = [
    # Models
    'CrystalGraphConvNet',
    'MaterialsTransformer', 
    'CatalystDesignNet',
    'PhasePredictor',
    'ElectronicStructureNet',
    
    # Predictors
    'PropertyPredictor',
    'StructurePredictor',
    'CatalystDesigner',
    'PhaseAnalyzer',
    'ElectronicStructureAnalyzer',
    
    # Data utilities
    'MaterialsDataset',
    'MaterialsDataLoader',
    'load_example_data',
    'generate_synthetic_materials_data',
    'load_materials_project_sample',
    'load_oqmd_sample',
    'create_crystal_graph',
    'save_materials_data',
    'load_materials_data',
    'compute_material_fingerprint',
    'create_materials_database',
    
    # Preprocessing
    'normalize_structure',
    'compute_structural_features',
    'compute_coordination_numbers',
    'compute_elemental_features',
    'augment_structure',
    'filter_structures',
    'standardize_features',
    'compute_radial_distribution_function',
    'compute_bond_angles',
    'detect_defects',
    
    # Visualization
    'plot_crystal_structure',
    'plot_property_distribution',
    'plot_property_correlation',
    'plot_phase_diagram',
    'plot_band_structure',
    'plot_catalyst_performance',
    'plot_structure_features',
    'plot_radial_distribution_function',
    'plot_formation_energy_vs_composition',
    'plot_prediction_results',
    'plot_materials_database_overview',
    'create_materials_report',
    'plot_defect_analysis'
]

# TODO: Implement materials science models and predictors
print("Materials module structure created. Implementation coming in future iterations.") 