# FSO Interplanetary Network Simulation

## Overview

This simulation models a Free Space Optical (FSO) communication network for interplanetary missions, featuring Earth-Mars connectivity through relay satellites positioned at Lagrange points. The system includes realistic orbital mechanics, traffic management, controller coordination, and solar conjunction resilience analysis.

## Quick Start

### Prerequisites
```bash
pip install numpy scipy networkx astropy spacepy matplotlib pandas
```

**For Figure Generation (additional):**
```bash
pip install matplotlib seaborn pandas numpy
```

### Running the Simulation
```python
python paste.py
```

The simulation will:
1. Initialize the satellite constellation
2. Run orbital propagation for ~2.25 years
3. Analyze network connectivity and performance
4. Export results to `results/` directory
5. Run validation tests for thesis analysis

## Configuration Constants

### Satellite Constellation
```python
EARTH_SATS = 24                    # Earth satellites in GEO
MARS_SATS = 8                      # Mars satellites in areosynchronous orbit
EARTH_MOON_CONTROLLERS = 4         # Controllers at Earth-Moon Lagrange points
EARTH_SUN_RELAYS = 4               # Earth-Sun relay satellites
MARS_SUN_RELAYS = 4                # Mars-Sun relay satellites
```

### Orbital Parameters
```python
EARTH_ORBIT_ALTITUDE = 35786.0     # km, Geostationary orbit
MARS_ORBIT_ALTITUDE = 17034.0      # km, Areostationary orbit
AU = 149597870.7                   # km, Astronomical Unit
```

### FSO Communication System
```python
FSO_WAVELENGTH = 1550e-9           # meters, Near-infrared
FSO_BEAM_DIVERGENCE = 1e-6         # radians, Extremely tight beam
FSO_TRANSMIT_POWER = 10.0          # Watts
FSO_APERTURE_DIAMETER = 0.3        # meters, Telescope aperture
FSO_DATA_RATE_BASELINE = 10e9      # bps, 10 Gbps baseline
```

### Simulation Parameters
```python
SIM_DURATION = 20000               # hours (~2.25 years)
SIM_STEP = 24.0                    # hours (1 day steps)
CONTROLLER_RANGE = 0.3 * AU        # km, Earth-Moon system coverage
RELAY_RANGE = 5.0 * AU             # km, Deep space relay range
```

### Network Management
```python
QOS_PRIORITY_LEVELS = ['EMERGENCY', 'CRITICAL', 'HIGH', 'NORMAL', 'LOW']
NAVIGATION_UPDATE_INTERVAL = 0.1   # hours (6 minutes)
TRAFFIC_UPDATE_INTERVAL = 0.05     # hours (3 minutes)
HANDOFF_THRESHOLD = 0.8            # Signal quality threshold
```

## Core Classes

### Satellite / EnhancedSatellite
**Purpose**: Represents individual satellites with orbital mechanics, FSO terminals, and coordination capabilities.

**Key Features**:
- Realistic orbital propagation with perturbations (J2, solar radiation pressure, third-body effects)
- FSO link establishment and management
- Controller coordination for Earth satellites
- Traffic scheduling and management

**Main Methods**:
- `update_position_with_orbital_perturbations()`: Updates satellite position using realistic physics
- `update_fso_links()`: Manages FSO link acquisition and maintenance
- `perform_controller_coordination()`: Coordinates regional satellite operations (controllers only)
- `_can_establish_fso_link()`: Determines FSO link feasibility with line-of-sight checking

### FSO_NetworkArchitecture
**Purpose**: Main simulation engine that manages the entire satellite constellation and network analysis.

**Key Features**:
- Network graph construction and connectivity analysis
- Performance metrics collection
- Comprehensive result export
- Validation test execution

**Main Methods**:
- `update()`: Main simulation step - updates all satellites and analyzes network
- `_analyze_connectivity()`: Calculates Earth-Mars connectivity percentages
- `_analyze_fso_performance()`: Analyzes FSO system performance metrics
- `export_fso_results()`: Exports comprehensive simulation results

### EnhancedFSO_Terminal
**Purpose**: Models FSO communication terminals with link budgets, acquisition, and traffic management.

**Key Features**:
- Realistic FSO link budget calculations
- Navigation-assisted pointing and acquisition
- Traffic coordination and scheduling
- Adaptive performance optimization

**Main Methods**:
- `attempt_link_acquisition()`: Establishes new FSO links
- `calculate_link_budget()`: Computes received power and link quality
- `receive_navigation_update()`: Processes controller guidance
- `is_authorized_to_transmit()`: Checks traffic coordination authorization

### RealEphemerisData
**Purpose**: Provides accurate planetary positions using NASA JPL ephemeris data.

**Key Features**:
- JPL DE440 ephemeris integration
- Earth, Mars, and Moon position calculations
- Time-accurate celestial mechanics

**Main Methods**:
- `get_earth_position()`: Returns Earth's position at given time
- `get_mars_position()`: Returns Mars's position at given time
- `get_moon_position()`: Returns Moon's position at given time

### PreciseLagrangePoints
**Purpose**: Calculates precise Lagrange point positions using actual mass ratios.

**Key Features**:
- All 5 Lagrange points for Earth-Moon, Earth-Sun, and Mars-Sun systems
- Real mass ratio calculations
- Time-varying positions

**Main Methods**:
- `earth_moon_l1()`, `earth_sun_l4()`, etc.: Calculate specific Lagrange point positions
- Supports all L1-L5 points for each system

### LineOfSightChecker
**Purpose**: Determines if direct optical communication is possible between satellites.

**Key Features**:
- Celestial body occlusion detection (Sun, Earth, Mars, Moon)
- Solar exclusion zone modeling
- Efficient geometric calculations

**Main Methods**:
- `is_line_of_sight_clear()`: Main LOS checking function
- `check_line_intersects_sphere()`: Geometric intersection calculation

### NavigationCoordinator
**Purpose**: Provides positioning guidance and traffic coordination for satellite regions.

**Key Features**:
- Optimal relay selection
- Trajectory prediction
- Handoff coordination
- Integrated traffic management

**Main Methods**:
- `generate_enhanced_coordination_update()`: Creates positioning + traffic instructions
- `coordinate_handoffs()`: Plans relay handoffs
- `select_optimal_relay()`: Chooses best relay for satellite

### TrafficManager
**Purpose**: Manages data transmission scheduling and load balancing across the network.

**Key Features**:
- Priority-based traffic queuing
- Transmission slot allocation
- Relay load balancing
- QoS enforcement

**Main Methods**:
- `simulate_traffic_demand()`: Models realistic traffic patterns
- `optimize_transmission_schedule()`: Creates efficient transmission schedules
- `coordinate_relay_load_balancing()`: Distributes load across relays

### RealisticOrbitalMechanics
**Purpose**: Implements high-fidelity orbital propagation with perturbations.

**Key Features**:
- J2 perturbation (Earth's oblateness)
- Solar radiation pressure
- Third-body gravitational effects
- Relativistic corrections

**Main Methods**:
- `propagate_orbit_with_perturbations()`: Main orbital integration function
- `predict_trajectory()`: Trajectory prediction for planning

### ThesisMetricsCollector
**Purpose**: Collects and exports comprehensive performance metrics for academic analysis.

**Key Features**:
- Time series data collection
- Performance indicator calculation
- Research-ready data export
- Validation test integration

**Main Methods**:
- `record_metrics()`: Records current network state
- `export_for_thesis()`: Exports data in thesis-ready formats

## Figure Generation Classes

### ThesisFigureGenerator
**Purpose**: Creates standard thesis figures from exported CSV data.

**Key Features**:
- Connectivity overview plots
- FSO performance analysis
- Solar conjunction visualization
- System comparison charts
- KPI summary dashboards

**Main Methods**:
- `create_connectivity_overview()`: Network connectivity over time
- `create_fso_performance_analysis()`: FSO system metrics
- `create_solar_conjunction_analysis()`: Blocking and conjunction effects
- `create_system_comparison_chart()`: Comparison with existing systems
- `generate_all_figures()`: Creates all basic figures

### EnhancedThesisFigureGenerator
**Purpose**: Extends basic generator with advanced technical figures.

**Key Features**:
- System architecture diagrams
- Design justification plots
- Performance validation figures
- Academic-quality comparative analysis

**Main Methods**:
- `create_system_architecture_diagram()`: Block diagram of network architecture
- `create_coverage_analysis()`: Planetary surface coverage
- `create_link_budget_analysis()`: FSO link budget breakdown
- `create_comprehensive_comparison()`: Academic comparison with citations
- `create_trade_off_analysis()`: Design trade-off visualization

### ValidationFigureGenerator
**Purpose**: Creates figures from validation test results.

**Key Features**:
- Validation test dashboards
- Recovery timeline visualization
- Traffic analysis plots
- Orbital configuration impact

**Main Methods**:
- `create_validation_summary_dashboard()`: Overall test results
- `create_recovery_timeline()`: Failure recovery analysis
- `create_traffic_analysis()`: Traffic overload performance
- `create_orbital_impact_visualization()`: Configuration impact analysis

### EnhancedThesisFigureGeneratorWithValidation
**Purpose**: Master class that coordinates all figure generation systems.

**Key Features**:
- Integrated figure generation pipeline
- Automatic data loading and validation
- Comprehensive error handling
- Publication-ready output

**Main Methods**:
- `generate_all_figures_including_validation()`: Creates complete figure set

## Output Files

### Time Series Data
- `thesis_data_time_series.csv`: Complete performance metrics over time
- `thesis_data_connectivity_comparison.csv`: Earth vs Mars connectivity analysis
- `thesis_data_fso_performance.csv`: FSO-specific performance metrics
- `thesis_data_blocking_analysis.csv`: Line-of-sight blocking statistics

### Summary Statistics
- `thesis_data_kpis.csv`: Key Performance Indicators summary
- `validation_summary.csv`: Validation test results
- `validation_test_results.json`: Detailed test data

### Generated Figures
After running the figure generator (`masters_figures.py`), the following publication-quality figures are created in the `figures/` directory:

**Basic Performance Figures:**
- `01_connectivity_overview.png`: Network connectivity over time
- `02_fso_performance.png`: FSO system performance metrics
- `03_solar_conjunction.png`: Solar conjunction analysis
- `04_system_comparison.png`: Comparison with current systems
- `05_kpi_summary.png`: Key performance indicators
- `06_thesis_summary.png`: Comprehensive thesis summary

**Advanced Architecture Figures:**
- `architecture_diagram.png`: System architecture block diagram
- `coverage_analysis.png`: Planetary surface coverage analysis
- `link_budget_analysis.png`: FSO link budget breakdown
- `latency_analysis.png`: Comprehensive latency characterization
- `system_comparison_cited.png`: Academic comparison with citations
- `trade_off_analysis.png`: Design trade-off analysis

**Validation Test Figures:**
- `validation_summary_dashboard.png`: All validation test results
- `orbital_configuration_analysis.png`: Orbital impact analysis
- `failure_recovery_timeline.png`: Network recovery timeline
- `traffic_overload_analysis.png`: Traffic handling performance

## Figure Generation System

### Prerequisites for Figure Generation
```bash
pip install matplotlib seaborn pandas numpy
```

### Running the Figure Generator
After completing the simulation, generate publication-quality figures:

```python
python masters_figures.py
```

The figure generator includes three integrated systems:

#### 1. Basic Thesis Figures (`ThesisFigureGenerator`)
Creates standard performance visualization figures:
- Network connectivity analysis
- FSO performance metrics
- Solar conjunction impact
- System comparisons
- KPI summaries

#### 2. Advanced Architecture Figures (`EnhancedThesisFigureGenerator`)
Generates detailed technical figures:
- System architecture diagrams with block representations
- Coverage analysis for Earth and Mars surface
- Link budget analysis with component breakdown
- Latency characterization across network segments
- Trade-off analysis (cost vs performance, complexity vs reliability)
- Academic comparison with proper citations

#### 3. Validation Test Figures (`ValidationFigureGenerator`)
Creates figures from validation test results:
- Comprehensive validation dashboard showing all test results
- Orbital configuration impact visualization
- Failure recovery timeline analysis
- Traffic overload performance metrics

### Figure Quality Settings
All figures are generated with publication quality:
- **DPI**: 300 (suitable for academic publications)
- **Format**: PNG with transparent backgrounds
- **Style**: Academic seaborn styling with professional color schemes
- **Fonts**: Publication-appropriate sizing and weights

### Customizing Figures
The figure generator can be customized by modifying:

```python
# Figure settings in masters_figures.py
fig_settings = {
    'dpi': 300,              # Increase for higher quality
    'bbox_inches': 'tight',  # Crop whitespace
    'facecolor': 'white',    # Background color
    'edgecolor': 'none'      # Border color
}

# Color schemes
colors = {
    'earth': '#4285f4',      # Earth system color
    'mars': '#ea4335',       # Mars system color
    'controller': '#34a853', # Controller color
    'relay': '#fbbc04',      # Relay color
    'link': '#666666'        # Link color
}
```

## Validation Tests

The simulation includes four validation tests:

1. **Solar Conjunction Test**: Verifies network maintains connectivity when Earth and Mars are on opposite sides of the Sun
2. **Dynamic Link Failure Recovery**: Tests network resilience to sudden link failures
3. **Traffic Overload Scenario**: Evaluates performance under heavy traffic loads
4. **Orbital Configuration Impact**: Analyzes performance across different planetary alignments

## Complete Workflow

### Full Analysis Pipeline
To run the complete analysis with figure generation:

```bash
# Step 1: Run the main simulation
python paste.py

# Step 2: Generate all figures
python masters_figures.py
```

### Directory Structure After Full Run
```
your-project/
├── paste.py                    # Main simulation script
├── masters_figures.py          # Figure generation script
├── results/                    # Generated data files
│   ├── thesis_data_*.csv      # Time series and analysis data
│   ├── validation_*.csv       # Validation test results
│   └── validation_test_results.json
└── figures/                   # Generated publication figures
    ├── 01_connectivity_overview.png
    ├── 02_fso_performance.png
    ├── architecture_diagram.png
    ├── validation_summary_dashboard.png
    └── [additional figures...]
```

### Troubleshooting Figure Generation

#### Common Issues

**1. Missing Data Files**
```
❌ Error: 'results' directory not found!
```
**Solution**: Run the main simulation first:
```bash
python paste.py
```

**2. Matplotlib/Seaborn Issues**
```
ImportError: No module named 'seaborn'
```
**Solution**: Install visualization dependencies:
```bash
pip install matplotlib seaborn pandas numpy
```

**3. Figure Quality Issues**
If figures appear blurry or pixelated:
```python
# In masters_figures.py, increase DPI
fig_settings = {
    'dpi': 600,  # Higher quality (larger file size)
    'bbox_inches': 'tight'
}
```

**4. Memory Issues with Large Datasets**
If the figure generator runs out of memory:
```python
# Reduce data sampling in figure generation
sample_rate = 10  # Use every 10th data point
data_subset = data[::sample_rate]
```

**5. Font/Style Issues**
If figures have font rendering problems:
```bash
# Install additional fonts (Linux/Mac)
sudo apt-get install fonts-dejavu-core  # Ubuntu
brew install font-dejavu                # macOS
```

### Performance Notes

- **Figure Generation Time**: ~2-5 minutes for complete set
- **Memory Usage**: ~1-2GB for large datasets
- **Disk Space**: ~50-100MB for all figures
- **Dependencies**: Matplotlib backend must support PNG output

## Extending the Model

### Adding New Satellite Types
1. Create new satellite instances in `FSO_NetworkArchitecture.__init__()`
2. Define orbital parameters and FSO terminal specifications
3. Update connectivity rules in `_can_establish_fso_link()`

### Modifying FSO Parameters
Adjust constants at the top of the file:
- `FSO_TRANSMIT_POWER`: Laser power in watts
- `FSO_APERTURE_DIAMETER`: Telescope aperture size
- `FSO_BEAM_DIVERGENCE`: Beam spread angle

### Custom Traffic Patterns
Modify `TrafficManager.simulate_traffic_demand()` to implement specific traffic scenarios or data generation patterns.

### New Validation Scenarios
Add custom tests to the validation suite by implementing new test functions and adding them to `run_selective_validation_tests()`.

### Adding Custom Figures
Create new figures by extending the figure generation classes:

```python
class CustomFigureGenerator(EnhancedThesisFigureGenerator):
    def create_custom_analysis(self):
        """Create your custom analysis figure"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Your custom plotting code here
        # Access data via self.time_series, self.kpis, etc.
        
        plt.savefig(f'{self.figures_dir}/custom_analysis.png', **self.fig_settings)
        plt.close()
    
    def generate_all_figures(self):
        """Override to include custom figures"""
        super().generate_all_figures()
        self.create_custom_analysis()
```

### Modifying Figure Appearance
Customize the visual style of figures:

```python
# In masters_figures.py
# Change color schemes
colors = {
    'earth': '#your_color',
    'mars': '#your_color',
    # ... other colors
}

# Modify plot styles
plt.style.use('your_preferred_style')
sns.set_palette("your_palette")

# Adjust figure sizes
figsize = (16, 12)  # Width, height in inches
```

## Performance Notes

- **Runtime**: ~15-45 minutes for full simulation depending on hardware
- **Memory**: Requires ~4-8GB RAM for large constellations
- **Disk**: Generates ~50-100MB of result data
- **CPU**: Benefits from multi-core processors for orbital calculations

## Academic Context

This simulation is designed for research on:
- Interplanetary communication network architectures
- Solar conjunction mitigation strategies
- FSO communication system performance
- Network resilience and traffic management
- Lagrange point utilization for space communications

### Publication-Ready Outputs
The system generates comprehensive academic outputs:

**Quantitative Analysis:**
- Time series performance data (CSV format)
- Statistical summaries and KPIs
- Validation test results with pass/fail criteria
- Comparative analysis with existing systems

**Visual Documentation:**
- System architecture diagrams suitable for publications
- Performance plots with academic styling
- Validation test dashboards
- Trade-off analysis visualizations

**Academic Citations:**
The figure generation system includes proper citations for:
- NASA DSN performance data
- ESA ground station specifications
- Academic papers on Mars communication systems
- Cost analysis references

### Research Applications
This model supports various research directions:

1. **Network Architecture Studies**: Modify constellation parameters to study optimal configurations
2. **Technology Assessment**: Compare FSO vs RF performance under various conditions
3. **Mission Planning**: Analyze communication windows and data return capabilities
4. **Economic Analysis**: Evaluate cost-benefit of different network architectures
5. **Reliability Studies**: Test network resilience under various failure scenarios

## File Reference

### Main Simulation Files
- `paste.py`: Primary simulation engine with all satellite classes and network architecture
- `masters_figures.py`: Comprehensive figure generation system

### Generated Data Files
- `results/thesis_data_*.csv`: Performance metrics and analysis data
- `results/validation_*.csv`: Validation test results
- `results/validation_test_results.json`: Complete test data in JSON format

### Generated Figure Files
- `figures/01-06_*.png`: Basic performance and analysis figures
- `figures/architecture_*.png`: System design and justification figures
- `figures/validation_*.png`: Validation test result visualizations

## License

[Add your license information here]