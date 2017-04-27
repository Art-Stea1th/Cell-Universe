using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Controls {

    using Extensions;
    using Helpers;

    public class MatrixLED : FrameworkElement {

        public byte[,] IntencitySource {
            get => (byte[,])GetValue(IntencitySourceProperty);
            set => SetValue(IntencitySourceProperty, value);
        }

        public Brush Background {
            get => (Brush)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        public Brush Foreground {
            get => (Brush)GetValue(ForegroundProperty);
            set => SetValue(ForegroundProperty, value);
        }

        public bool ShowGrid {
            get => (bool)GetValue(ShowGridProperty);
            set => SetValue(ShowGridProperty, value);
        }

        public static readonly DependencyProperty IntencitySourceProperty = DependencyProperty.Register(
            "IntencitySource", typeof(byte[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender, OnIntencitySourceChanged));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits));

        public static readonly DependencyProperty ShowGridProperty = DependencyProperty.Register(
            "ShowGrid", typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                true, FrameworkPropertyMetadataOptions.AffectsRender));

        static MatrixLED() {
            Style style = CreateDefaultStyles();
            StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(style));
        }

        private static Style CreateDefaultStyles() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        private static void OnIntencitySourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            var matrix = d as MatrixLED;
            matrix.ledsNext = e.NewValue as byte[,];
            matrix.UpdateBuffer();
            matrix.ledsPrev = matrix.ledsNext;
        }

        private void UpdateBuffer() {

            if (ledsPrev == null) {
                if (ledsNext == null) {
                    return;
                }
                else {
                    RecalculateMaskSize();
                    NewGridMask();
                    NewLedsMask();
                }
            }
            else {
                if (ledsNext == null) {
                    ClearLedsMask();
                }
                else {
                    if (SizeEquals(ledsPrev, ledsNext)) {
                        RepaintLedsMask();
                    }
                    else {
                        RecalculateMaskSize();
                        NewGridMask();
                        NewLedsMask();
                    }
                }
            }
        }

        private static bool SizeEquals<T>(T[,] array1, T[,] array2)
            => array1.GetLength(0) == array2.GetLength(0) && array1.GetLength(1) == array2.GetLength(1);

        private void RecalculateMaskSize() {
            var cellsSizeX = ledsNext.CountX() * cellSize;
            var cellsSizeY = ledsNext.CountY() * cellSize;
            var thicknessSizeX = ledsNext.CountX() * thickness + thickness;
            var thisknessSizeY = ledsNext.CountY() * thickness + thickness;
            contentSize = new IntPair(cellsSizeX + thicknessSizeX, cellsSizeY + thisknessSizeY);
        }

        private void NewGridMask() {
            gridBitmap = BitmapHelper.CreateWriteable(contentSize.X, contentSize.Y);
            RepaintGridMask();
        }

        private void NewLedsMask() {
            ledsBitmap = BitmapHelper.CreateWriteable(contentSize.X, contentSize.Y);
            RepaintLedsMask();
        }

        private void RepaintGridMask() {            
            using (var context = new WriteableContext(gridBitmap)) {
                context.WriteGrid(cellSize, thickness, Color.FromArgb(10, 255, 255, 255));
            }
        }

        private void RepaintLedsMask() {
            using (var context = new WriteableContext(ledsBitmap)) {
                context.WriteRectSequence(GetChangedLeds()
                    .Select(l => (
                    x: thickness + l.x * thickness + l.x * cellSize,
                    y: thickness + l.y * thickness + l.y * cellSize,
                    color: l.newValue == 0 ? solidColor : emptyColor)
                    ),
                    cellSize);
            }
        }

        private void ClearLedsMask() => ledsBitmap = null;

        private IEnumerable<(int x, int y, byte newValue)> GetChangedLeds() {
            if (ledsPrev == null) {
                return ledsNext.AsEnumerableIndexed();
            }
            return ledsPrev.AsEnumerableIndexed().Except(ledsNext.AsEnumerableIndexed());
        }
        
        
        // EXPERIMENTS

        protected override void OnRender(DrawingContext dc) {

            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));

            dc.PushOpacityMask(LedsOpacityMask);

            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));

            dc.Pop();
            

            dc.DrawImage(gridBitmap, new Rect(new Point(), RenderSize));
        }

        // END EXPERIMENTS

        protected override Size MeasureOverride(Size constraint) {
            return ComputeSize(constraint);
        }

        protected override Size ArrangeOverride(Size arrangeSize) {
            return ComputeSize(arrangeSize);
        }

        private Size ComputeSize(Size availableSize) {
            return MeasureArrangeHelper.ComputeSize(availableSize, contentSize);
        }

        private Color solidColor = Color.FromArgb(255, 0, 0, 0);
        private Color emptyColor = Color.FromArgb(0, 0, 0, 0);

        private int cellSize = 4;
        private int thickness = 1;

        private ImageBrush GridOpacityMask => new ImageBrush(gridBitmap);
        private WriteableBitmap gridBitmap;

        private ImageBrush LedsOpacityMask => new ImageBrush(ledsBitmap);
        private WriteableBitmap ledsBitmap;

        private byte[,] ledsPrev;
        private byte[,] ledsNext;

        private IntPair contentSize;

        private struct IntPair {

            public int X { get; }
            public int Y { get; }

            public IntPair(int x, int y) { X = x; Y = y; }

            public static explicit operator IntPair(Size size)
                => new IntPair((int)size.Width, (int)size.Height);

            public static explicit operator IntPair(Int32Rect rect)
                => new IntPair(rect.Width, rect.Height);

            public static implicit operator Size(IntPair pair)
                => new Size(pair.X, pair.Y);

            public static implicit operator Int32Rect(IntPair pair)
                => new Int32Rect(0, 0, pair.X, pair.Y);
        }
    }
}