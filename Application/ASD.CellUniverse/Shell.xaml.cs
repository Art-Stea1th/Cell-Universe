using System.Windows;
using Microsoft.Practices.Unity;
using Prism.Regions;

namespace ASD.CellUniverse {

    using Infrastructure.Properties;
    using Views;

    public partial class Shell : Window {

        IUnityContainer container;
        IRegionManager regionManager;

        public Shell(IUnityContainer container, IRegionManager regionManager) {

            InitializeComponent();

            this.container = container;
            this.regionManager = regionManager;

            SourceInitialized += (s, e) => InitializeRegions();
        }

        private void InitializeRegions() {

            regionManager.Regions[Settings.Default.ShellRegionMainName].Add(container.Resolve<CellSurfaceView>());
            regionManager.Regions[Settings.Default.ShellRegionPanelName].Add(container.Resolve<ControlsPanelView>());

        }
    }
}