using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace RCVL {

    [TemplatePart(Name = CellularViewport.CellSurfaceControl, Type = typeof(Image))]
    public class CellularViewport : Control {

        private const string CellSurfaceControl = "PART_CellSurfaceControl";

        private Image  _cellSurfaceControl;

        public static readonly DependencyProperty CellularDataProperty;

        public List<bool[,]> CellularData {
            get { return (List<bool[,]>)GetValue(CellularDataProperty); }
            set { SetValue(CellularDataProperty, value); }
        }

        static CellularViewport() {
            DefaultStyleKeyProperty.OverrideMetadata(
                typeof(CellularViewport), new FrameworkPropertyMetadata(typeof(CellularViewport)));

            CellularDataProperty = DependencyProperty.Register(
                "CellularData", typeof(List<bool[,]>), typeof(CellularViewport),
                new FrameworkPropertyMetadata(new List<bool[,]> { new bool[1, 1] { { true } } }));
        }

        internal WriteableBitmap CellSurface { get; private set; }

        public override void OnApplyTemplate() {
            base.OnApplyTemplate();
            _cellSurfaceControl = GetTemplateChild(CellSurfaceControl) as Image;
            CorrectParameters();
        }

        private void CorrectParameters() {
            MinWidth = MinWidth < 1 ? 1 : MinWidth;
            MinHeight = MinHeight < 1 ? 1 : MinHeight;
        }

        protected override void OnRender(DrawingContext drawingContext) {
            base.OnRender(drawingContext);

            ReinitializeData();

            // --- Test Fill ---
            DrawRect(CellSurface, 0, 0, (int)ActualWidth, (int)ActualHeight, GetIntColor(0, 128, 255));
        }

        private void ReinitializeData() {
            CellSurface = new WriteableBitmap(
                    (int)ActualWidth,
                    (int)ActualHeight, 96, 96, PixelFormats.Bgr32, null);
            _cellSurfaceControl.Source = CellSurface;
        }

        private void DrawRect(WriteableBitmap targetBitmap, int startPosX, int startPosY, int width, int height, int color) {

            if (startPosX + width > (int)ActualWidth || startPosY + height > (int)ActualHeight) {
                return;
            }

            targetBitmap.Lock();
            unsafe
            {
                for (int y = startPosY; y < startPosY + height; ++y) {
                    for (int x = startPosX; x < startPosX + width; ++x) {

                        int pBackBuffer = (int)CellSurface.BackBuffer;

                        pBackBuffer += y * targetBitmap.BackBufferStride;
                        pBackBuffer += x * 4;

                        *((int*)pBackBuffer) = color;
                    }
                }
                targetBitmap.AddDirtyRect(new Int32Rect(startPosX, startPosY, width, height));
            }
            targetBitmap.Unlock();
        }

        private int GetIntColor(byte red, byte green, byte blue) {
            int result = red << 16;
            result |= green << 8;
            result |= blue << 0;
            return result;
        }

        public CellularViewport() { }

    }
}