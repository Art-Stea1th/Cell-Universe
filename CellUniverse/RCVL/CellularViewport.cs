using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;


namespace RCVL {

    [TemplatePart(Name = CellularViewport.CellSurfaceControl, Type = typeof(Image))]
    public sealed class CellularViewport : Control {

        #region DependencyProperties

        public static readonly DependencyProperty CellularDataProperty;

        public Color[,] CellularData {
            get { return (Color[,])GetValue(CellularDataProperty); }
            set { SetValue(CellularDataProperty, value); }
        }

        static CellularViewport() {

            DefaultStyleKeyProperty.OverrideMetadata(
                typeof(CellularViewport), new FrameworkPropertyMetadata(typeof(CellularViewport)));

            var cellularDataPropertyMetadata = new FrameworkPropertyMetadata();

            cellularDataPropertyMetadata.DefaultValue         = null;
            cellularDataPropertyMetadata.BindsTwoWayByDefault = true;
            cellularDataPropertyMetadata.AffectsRender        = true;

            cellularDataPropertyMetadata.PropertyChangedCallback = OnCellularDataChangedCallback;

            CellularDataProperty =
                DependencyProperty.Register(
                    "CellularData", typeof(Color[,]), typeof(CellularViewport), cellularDataPropertyMetadata);                        
        }

        private static void OnCellularDataChangedCallback(DependencyObject sender, DependencyPropertyChangedEventArgs e) {

            CellularViewport cellViewport = (CellularViewport)sender;
            cellViewport.CellularData     = (Color[,])e.NewValue;
            cellViewport.oldCellularData  = (Color[,])e.OldValue;
        }

        #endregion

        #region TemplateParts

        private const string CellSurfaceControl = "PART_CellSurfaceControl";
        private Image  cellSurfaceControl;

        #endregion

        private Color[,] oldCellularData;

        private ViewportSettings settings;
        private ViewportRenderer renderer;

        public override void OnApplyTemplate() {

            cellSurfaceControl = GetTemplateChild(CellSurfaceControl) as Image;
            CoerceParameters();

            settings = new ViewportSettings();
            renderer = new ViewportRenderer(settings);
        }

        private void CoerceParameters() {
            MinWidth  = GetNonZeroGuaranteed(MinWidth);
            MinHeight = GetNonZeroGuaranteed(MinHeight);
        }

        protected override void OnRender(DrawingContext drawingContext) {

            if (CellularData == null) {
                return;
            }

            int surfaceWidth    = GetNonZeroGuaranteed(ActualWidth);
            int surfaceHeight   = GetNonZeroGuaranteed(ActualHeight);
            int cellsHorizontal = GetNonZeroGuaranteed(CellularData.GetLength(1));
            int cellsVertical   = GetNonZeroGuaranteed(CellularData.GetLength(0));

            renderer.Update(surfaceWidth, surfaceHeight, cellsHorizontal, cellsVertical);
            cellSurfaceControl.Source = renderer.Render(oldCellularData, CellularData);
        }

        private int GetNonZeroGuaranteed(double value) {
            return value > 0 ? (int)value : 1;
        }
    }
}