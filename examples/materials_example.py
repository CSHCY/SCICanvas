"""
Comprehensive Materials Science Example

This script demonstrates the materials science capabilities of SCICanvas,
including:
1. Property prediction (formation energy, band gap, etc.)
2. Crystal structure prediction
3. Catalyst design and optimization
4. Phase diagram analysis
5. Electronic structure prediction
6. Comprehensive visualization and analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import scicanvas
sys.path.append(str(Path(__file__).parent.parent))

try:
    from scicanvas.materials import (
        # Data utilities
        load_example_data,
        generate_synthetic_materials_data,
        MaterialsDataLoader,
        
        # Predictors
        PropertyPredictor,
        StructurePredictor,
        CatalystDesigner,
        PhaseAnalyzer,
        ElectronicStructureAnalyzer,
        
        # Preprocessing
        normalize_structure,
        compute_structural_features,
        compute_elemental_features,
        augment_structure,
        filter_structures,
        detect_defects,
        
        # Visualization
        plot_crystal_structure,
        plot_property_distribution,
        plot_property_correlation,
        plot_phase_diagram,
        plot_band_structure,
        plot_catalyst_performance,
        plot_prediction_results,
        create_materials_report
    )
    
    print("✓ Successfully imported SCICanvas materials module")
    
except ImportError as e:
    print(f"✗ Error importing SCICanvas materials module: {e}")
    print("Please ensure SCICanvas is properly installed")
    sys.exit(1)


def demonstrate_property_prediction():
    """Demonstrate materials property prediction."""
    print("\n" + "="*60)
    print("MATERIALS PROPERTY PREDICTION DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate synthetic materials data
        print("Generating synthetic materials data...")
        data = generate_synthetic_materials_data(n_materials=50, max_atoms=30)
        
        structures = data['structures']
        properties = data['properties']
        
        print(f"Generated {len(structures)} materials with properties")
        
        # Extract formation energies for prediction
        formation_energies = [p['formation_energy'] for p in properties]
        
        print("\n1. Formation Energy Prediction")
        print("-" * 40)
        
        # Initialize property predictor
        predictor = PropertyPredictor(
            model_type='cgcnn',
            property_type='formation_energy',
            device='cpu'
        )
        
        # Train the model
        print("Training formation energy predictor...")
        predictor.fit(
            structures=structures[:40],  # Use first 40 for training
            properties=formation_energies[:40],
            batch_size=8,
            num_epochs=20,
            learning_rate=1e-3
        )
        
        # Make predictions on test set
        print("Making predictions on test set...")
        test_structures = structures[40:]
        test_energies = formation_energies[40:]
        
        predicted_energies = []
        for structure in test_structures:
            pred = predictor.predict(structure)
            predicted_energies.append(pred)
        
        print(f"Predicted {len(predicted_energies)} formation energies")
        
        # Visualize results
        print("Visualizing prediction results...")
        plot_prediction_results(
            true_values=test_energies,
            predicted_values=predicted_energies,
            property_name="Formation Energy (eV/atom)"
        )
        
        # Calculate metrics
        mse = np.mean((np.array(test_energies) - np.array(predicted_energies))**2)
        mae = np.mean(np.abs(np.array(test_energies) - np.array(predicted_energies)))
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        print("\n2. Band Gap Prediction")
        print("-" * 40)
        
        # Band gap prediction
        band_gaps = [p['band_gap'] for p in properties]
        
        bg_predictor = PropertyPredictor(
            model_type='transformer',
            property_type='band_gap',
            device='cpu'
        )
        
        print("Training band gap predictor...")
        bg_predictor.fit(
            structures=structures[:40],
            properties=band_gaps[:40],
            batch_size=8,
            num_epochs=15,
            learning_rate=1e-3
        )
        
        # Predict band gaps
        predicted_bg = bg_predictor.predict_batch(test_structures)
        
        print("Visualizing band gap predictions...")
        plot_prediction_results(
            true_values=band_gaps[40:],
            predicted_values=predicted_bg,
            property_name="Band Gap (eV)"
        )
        
        return {
            'formation_energy_predictor': predictor,
            'band_gap_predictor': bg_predictor,
            'test_results': {
                'formation_energy': (test_energies, predicted_energies),
                'band_gap': (band_gaps[40:], predicted_bg)
            }
        }
        
    except Exception as e:
        print(f"Error in property prediction demonstration: {e}")
        return None


def demonstrate_structure_prediction():
    """Demonstrate crystal structure prediction."""
    print("\n" + "="*60)
    print("CRYSTAL STRUCTURE PREDICTION DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate synthetic data
        print("Generating synthetic structure data...")
        data = generate_synthetic_materials_data(n_materials=30, max_atoms=20)
        
        structures = data['structures']
        compositions = data['compositions']
        
        print(f"Generated {len(structures)} structures for training")
        
        # Initialize structure predictor
        predictor = StructurePredictor(
            model_type='transformer',
            max_atoms=20,
            device='cpu'
        )
        
        # Prepare training data
        train_compositions = []
        train_structures = []
        
        for i, (structure, composition) in enumerate(zip(structures[:20], compositions[:20])):
            # Convert composition to atom types
            atom_types = structure['atom_types']
            train_compositions.append(atom_types)
            train_structures.append(structure)
        
        print("Training structure predictor...")
        predictor.fit(
            compositions=train_compositions,
            structures=train_structures,
            batch_size=4,
            num_epochs=15,
            learning_rate=1e-4
        )
        
        # Predict structures for test compositions
        print("Predicting crystal structures...")
        test_compositions = [structures[i]['atom_types'] for i in range(20, 25)]
        
        predicted_structures = []
        for composition in test_compositions:
            pred_structure = predictor.predict(composition)
            predicted_structures.append(pred_structure)
        
        print(f"Predicted {len(predicted_structures)} crystal structures")
        
        # Visualize some predicted structures
        print("Visualizing predicted structures...")
        for i, pred_struct in enumerate(predicted_structures[:2]):
            print(f"Plotting predicted structure {i+1}...")
            
            # Create a proper structure dictionary for visualization
            viz_structure = {
                'coordinates': pred_struct['coordinates'],
                'atom_types': pred_struct['atom_types'],
                'n_atoms': pred_struct['n_atoms']
            }
            
            plot_crystal_structure(viz_structure, show_bonds=False)
        
        return {
            'structure_predictor': predictor,
            'predicted_structures': predicted_structures
        }
        
    except Exception as e:
        print(f"Error in structure prediction demonstration: {e}")
        return None


def demonstrate_catalyst_design():
    """Demonstrate catalyst design and optimization."""
    print("\n" + "="*60)
    print("CATALYST DESIGN AND OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate synthetic catalyst data
        print("Generating synthetic catalyst data...")
        data = generate_synthetic_materials_data(n_materials=40, max_atoms=25)
        
        catalyst_structures = data['catalyst_structures']
        reaction_conditions = data['reaction_conditions']
        activities = data['activities']
        selectivities = data['selectivities']
        
        print(f"Generated {len(catalyst_structures)} catalyst examples")
        
        # Initialize catalyst designer
        designer = CatalystDesigner(device='cpu')
        
        print("Training catalyst design model...")
        designer.fit(
            catalyst_structures=catalyst_structures[:15],
            reaction_conditions=reaction_conditions[:15],
            activities=activities[:15],
            selectivities=selectivities[:15],
            batch_size=4,
            num_epochs=20,
            learning_rate=1e-3
        )
        
        # Predict catalyst performance
        print("Predicting catalyst performance...")
        test_catalysts = catalyst_structures[15:]
        test_conditions = reaction_conditions[15:]
        
        predicted_performance = []
        for catalyst, conditions in zip(test_catalysts, test_conditions):
            performance = designer.predict(catalyst, conditions)
            predicted_performance.append(performance)
        
        # Extract predicted activities and selectivities
        pred_activities = [p['activity'] for p in predicted_performance]
        pred_selectivities = [p['selectivity'] for p in predicted_performance]
        
        print(f"Predicted performance for {len(predicted_performance)} catalysts")
        
        # Visualize catalyst performance
        print("Visualizing catalyst performance...")
        plot_catalyst_performance(
            activities=pred_activities,
            selectivities=pred_selectivities,
            catalyst_names=[f"Catalyst {i+1}" for i in range(len(pred_activities))]
        )
        
        # Compare with true values
        true_activities = activities[15:]
        true_selectivities = selectivities[15:]
        
        print("Activity prediction results:")
        plot_prediction_results(
            true_values=true_activities,
            predicted_values=pred_activities,
            property_name="Catalytic Activity"
        )
        
        print("Selectivity prediction results:")
        plot_prediction_results(
            true_values=true_selectivities,
            predicted_values=pred_selectivities,
            property_name="Selectivity"
        )
        
        return {
            'catalyst_designer': designer,
            'performance_predictions': predicted_performance
        }
        
    except Exception as e:
        print(f"Error in catalyst design demonstration: {e}")
        return None


def demonstrate_phase_analysis():
    """Demonstrate phase diagram analysis."""
    print("\n" + "="*60)
    print("PHASE DIAGRAM ANALYSIS DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate synthetic phase data
        print("Generating synthetic phase data...")
        data = generate_synthetic_materials_data(n_materials=60, max_atoms=20)
        
        compositions = data['compositions']
        conditions = data['conditions']
        phases = data['phases']
        stabilities = data['stabilities']
        
        print(f"Generated {len(compositions)} phase data points")
        
        # Initialize phase analyzer
        analyzer = PhaseAnalyzer(n_phases=5, device='cpu')
        
        print("Training phase prediction model...")
        analyzer.fit(
            compositions=compositions[:45],
            conditions=conditions[:45],
            phases=phases[:45],
            stabilities=stabilities[:45],
            batch_size=8,
            num_epochs=25,
            learning_rate=1e-3
        )
        
        # Predict phases for test data
        print("Predicting phase stability...")
        test_compositions = compositions[45:]
        test_conditions = conditions[45:]
        test_phases = phases[45:]
        
        phase_predictions = []
        for comp, cond in zip(test_compositions, test_conditions):
            prediction = analyzer.predict(comp, cond)
            phase_predictions.append(prediction)
        
        # Extract predicted phases
        pred_phases = [p['predicted_phase'] for p in phase_predictions]
        pred_stabilities = [p['stability_score'] for p in phase_predictions]
        
        print(f"Predicted phases for {len(phase_predictions)} compositions")
        
        # Visualize phase diagram
        print("Visualizing phase diagram...")
        plot_phase_diagram(
            compositions=test_compositions,
            phases=pred_phases
        )
        
        # Calculate phase prediction accuracy
        accuracy = np.mean(np.array(pred_phases) == np.array(test_phases))
        print(f"Phase prediction accuracy: {accuracy:.3f}")
        
        # Visualize stability predictions
        plot_prediction_results(
            true_values=stabilities[45:],
            predicted_values=pred_stabilities,
            property_name="Phase Stability"
        )
        
        return {
            'phase_analyzer': analyzer,
            'phase_predictions': phase_predictions
        }
        
    except Exception as e:
        print(f"Error in phase analysis demonstration: {e}")
        return None


def demonstrate_electronic_structure():
    """Demonstrate electronic structure prediction."""
    print("\n" + "="*60)
    print("ELECTRONIC STRUCTURE PREDICTION DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate synthetic electronic structure data
        print("Generating synthetic electronic structure data...")
        data = generate_synthetic_materials_data(n_materials=40, max_atoms=20)
        
        structures = data['structures']
        properties = data['properties']
        electronic_data = data['electronic_data']
        
        # Extract electronic properties
        band_gaps = [p['band_gap'] for p in properties]
        dos_spectra = [e['dos_spectrum'] for e in electronic_data]
        fermi_levels = [e['fermi_level'] for e in electronic_data]
        
        print(f"Generated electronic structure data for {len(structures)} materials")
        
        # Initialize electronic structure analyzer
        analyzer = ElectronicStructureAnalyzer(device='cpu')
        
        print("Training electronic structure model...")
        analyzer.fit(
            structures=structures[:25],
            band_gaps=band_gaps[:25],
            dos_spectra=dos_spectra[:25],
            fermi_levels=fermi_levels[:25],
            batch_size=4,
            num_epochs=20,
            learning_rate=1e-3
        )
        
        # Predict electronic properties
        print("Predicting electronic structure properties...")
        test_structures = structures[25:30]
        
        electronic_predictions = []
        for structure in test_structures:
            prediction = analyzer.predict(structure)
            electronic_predictions.append(prediction)
        
        # Extract predictions
        pred_band_gaps = [p['band_gap'] for p in electronic_predictions]
        pred_fermi_levels = [p['fermi_level'] for p in electronic_predictions]
        
        print(f"Predicted electronic properties for {len(electronic_predictions)} materials")
        
        # Visualize band structure
        print("Visualizing electronic structure...")
        plot_band_structure(
            band_gaps=pred_band_gaps,
            dos_spectra=[p['dos_spectrum'] for p in electronic_predictions]
        )
        
        # Compare predictions with true values
        true_band_gaps = band_gaps[25:30]
        true_fermi_levels = fermi_levels[25:30]
        
        print("Band gap prediction results:")
        plot_prediction_results(
            true_values=true_band_gaps,
            predicted_values=pred_band_gaps,
            property_name="Band Gap (eV)"
        )
        
        print("Fermi level prediction results:")
        plot_prediction_results(
            true_values=true_fermi_levels,
            predicted_values=pred_fermi_levels,
            property_name="Fermi Level (eV)"
        )
        
        # Classify materials
        print("\nMaterial Classification:")
        for i, prediction in enumerate(electronic_predictions):
            material_type = "Metal" if prediction['is_metal'] else \
                           "Semiconductor" if prediction['is_semiconductor'] else \
                           "Insulator"
            print(f"Material {i+1}: {material_type} (Band gap: {prediction['band_gap']:.2f} eV)")
        
        return {
            'electronic_analyzer': analyzer,
            'electronic_predictions': electronic_predictions
        }
        
    except Exception as e:
        print(f"Error in electronic structure demonstration: {e}")
        return None


def demonstrate_preprocessing_and_analysis():
    """Demonstrate preprocessing and structural analysis."""
    print("\n" + "="*60)
    print("PREPROCESSING AND STRUCTURAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate synthetic data
        print("Generating synthetic materials data...")
        data = generate_synthetic_materials_data(n_materials=20, max_atoms=25)
        structures = data['structures']
        
        print(f"Analyzing {len(structures)} crystal structures")
        
        print("\n1. Structure Normalization")
        print("-" * 40)
        
        # Normalize structures
        normalized_structures = []
        for structure in structures[:5]:
            normalized = normalize_structure(structure)
            normalized_structures.append(normalized)
            print(f"Normalized structure with {len(structure['atom_types'])} atoms")
        
        print("\n2. Structural Feature Extraction")
        print("-" * 40)
        
        # Compute structural features
        structural_features = []
        for structure in structures[:10]:
            features = compute_structural_features(structure)
            structural_features.append(features)
        
        # Display some features
        feature_names = list(structural_features[0].keys())[:5]
        print(f"Extracted {len(feature_names)} structural features:")
        for name in feature_names:
            values = [f[name] for f in structural_features]
            print(f"  {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
        
        print("\n3. Elemental Analysis")
        print("-" * 40)
        
        # Compute elemental features
        elemental_features = []
        for structure in structures[:10]:
            features = compute_elemental_features(structure['atom_types'])
            elemental_features.append(features)
        
        # Display elemental features
        elem_feature_names = list(elemental_features[0].keys())[:3]
        print(f"Extracted {len(elem_feature_names)} elemental features:")
        for name in elem_feature_names:
            values = [f[name] for f in elemental_features]
            print(f"  {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
        
        print("\n4. Data Augmentation")
        print("-" * 40)
        
        # Demonstrate data augmentation
        original_structure = structures[0]
        
        # Rotation augmentation
        rotated = augment_structure(original_structure, 'rotation')
        print("Applied rotation augmentation")
        
        # Noise augmentation
        noisy = augment_structure(original_structure, 'noise', noise_std=0.05)
        print("Applied noise augmentation")
        
        # Scaling augmentation
        scaled = augment_structure(original_structure, 'scaling', scale_factor=1.1)
        print("Applied scaling augmentation")
        
        print("\n5. Defect Detection")
        print("-" * 40)
        
        # Detect defects in structures
        for i, structure in enumerate(structures[:3]):
            defects = detect_defects(structure)
            total_defects = defects['summary']['total_defects']
            defect_density = defects['summary']['defect_density']
            
            print(f"Structure {i+1}: {total_defects} defects, density={defect_density:.3f}")
        
        print("\n6. Structure Filtering")
        print("-" * 40)
        
        # Filter structures based on quality criteria
        filtered_structures, valid_indices = filter_structures(
            structures,
            min_atoms=5,
            max_atoms=40,
            min_distance=0.5,
            max_distance=15.0
        )
        
        print(f"Filtered {len(structures)} → {len(filtered_structures)} structures")
        print(f"Removed {len(structures) - len(filtered_structures)} low-quality structures")
        
        return {
            'normalized_structures': normalized_structures,
            'structural_features': structural_features,
            'elemental_features': elemental_features,
            'filtered_structures': filtered_structures
        }
        
    except Exception as e:
        print(f"Error in preprocessing demonstration: {e}")
        return None


def create_comprehensive_report():
    """Create a comprehensive materials analysis report."""
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE MATERIALS REPORT")
    print("="*60)
    
    try:
        # Generate comprehensive dataset
        print("Generating comprehensive materials dataset...")
        data = generate_synthetic_materials_data(n_materials=50, max_atoms=30)
        
        structures = data['structures']
        properties = data['properties']
        
        # Create report directory
        report_dir = "materials_analysis_report"
        
        print(f"Creating comprehensive report in {report_dir}/")
        
        # Generate the report
        create_materials_report(
            structures=structures,
            properties=properties,
            save_dir=report_dir
        )
        
        print("✓ Comprehensive materials report generated successfully!")
        print(f"  Check the '{report_dir}' directory for all plots and analysis")
        
        return True
        
    except Exception as e:
        print(f"Error creating comprehensive report: {e}")
        return False


def main():
    """Main function to run all demonstrations."""
    print("🚀 SCICanvas Materials Science Comprehensive Demonstration")
    print("=" * 80)
    
    # Store results from each demonstration
    results = {}
    
    # 1. Property Prediction
    prop_results = demonstrate_property_prediction()
    if prop_results:
        results['property_prediction'] = prop_results
        print("✓ Property prediction demonstration completed successfully")
    else:
        print("✗ Property prediction demonstration failed")
    
    # 2. Structure Prediction
    struct_results = demonstrate_structure_prediction()
    if struct_results:
        results['structure_prediction'] = struct_results
        print("✓ Structure prediction demonstration completed successfully")
    else:
        print("✗ Structure prediction demonstration failed")
    
    # 3. Catalyst Design
    catalyst_results = demonstrate_catalyst_design()
    if catalyst_results:
        results['catalyst_design'] = catalyst_results
        print("✓ Catalyst design demonstration completed successfully")
    else:
        print("✗ Catalyst design demonstration failed")
    
    # 4. Phase Analysis
    phase_results = demonstrate_phase_analysis()
    if phase_results:
        results['phase_analysis'] = phase_results
        print("✓ Phase analysis demonstration completed successfully")
    else:
        print("✗ Phase analysis demonstration failed")
    
    # 5. Electronic Structure
    electronic_results = demonstrate_electronic_structure()
    if electronic_results:
        results['electronic_structure'] = electronic_results
        print("✓ Electronic structure demonstration completed successfully")
    else:
        print("✗ Electronic structure demonstration failed")
    
    # 6. Preprocessing and Analysis
    preprocessing_results = demonstrate_preprocessing_and_analysis()
    if preprocessing_results:
        results['preprocessing'] = preprocessing_results
        print("✓ Preprocessing and analysis demonstration completed successfully")
    else:
        print("✗ Preprocessing and analysis demonstration failed")
    
    # 7. Comprehensive Report
    report_success = create_comprehensive_report()
    if report_success:
        print("✓ Comprehensive report generation completed successfully")
    else:
        print("✗ Comprehensive report generation failed")
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    
    successful_demos = len([r for r in results.values() if r is not None])
    total_demos = 6  # Number of main demonstrations
    
    print(f"Successfully completed: {successful_demos}/{total_demos} demonstrations")
    
    if successful_demos == total_demos:
        print("🎉 All materials science demonstrations completed successfully!")
        print("\nThe SCICanvas materials science module provides:")
        print("  ✓ Property prediction (formation energy, band gap, etc.)")
        print("  ✓ Crystal structure prediction from composition")
        print("  ✓ Catalyst design and optimization")
        print("  ✓ Phase diagram analysis and prediction")
        print("  ✓ Electronic structure prediction")
        print("  ✓ Comprehensive preprocessing and analysis tools")
        print("  ✓ Advanced visualization capabilities")
        print("  ✓ Automated report generation")
    else:
        print("⚠️  Some demonstrations encountered issues.")
        print("This may be due to missing dependencies or system limitations.")
    
    print("\n📊 Materials Science Module Features:")
    print("  • 5 neural network models (CGCNN, Transformer, etc.)")
    print("  • 5 specialized predictors for different tasks")
    print("  • Comprehensive data handling and preprocessing")
    print("  • Advanced visualization and analysis tools")
    print("  • Integration with materials databases")
    print("  • Production-ready implementations")
    
    return results


if __name__ == "__main__":
    # Set matplotlib backend for headless environments
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except:
        pass
    
    # Run the comprehensive demonstration
    results = main() 