using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Controls {

    using Helpers;

    public class MatrixLED : FrameworkElement {

        private Grid grid;
        private WriteableBitmap mask;

        public uint[,] Source { get => (uint[,])GetValue(SourceProperty); set => SetValue(SourceProperty, value); }

        public Brush Background { get => (Brush)GetValue(BackgroundProperty); set => SetValue(BackgroundProperty, value); }
        public Brush Foreground { get => (Brush)GetValue(ForegroundProperty); set => SetValue(ForegroundProperty, value); }

        public bool SplitCells { get => (bool)GetValue(SplitCellsProperty); set => SetValue(SplitCellsProperty, value); }


        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(nameof(
            Source), typeof(uint[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender, OnMatrixChanged));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty SplitCellsProperty = DependencyProperty.Register(nameof(
            SplitCells), typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                false, FrameworkPropertyMetadataOptions.AffectsRender, OnSplitCellsChanged));

        static MatrixLED() {
            StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(CreateStyle()));
        }

        private static Style CreateStyle() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        public MatrixLED() => grid = new Grid();

        private static void OnMatrixChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            var matrix = d as MatrixLED;
            var newSource = e.NewValue as uint[,];

            if (newSource == null || newSource.Length < 1) {
                matrix.grid.RecalculateContentSize(1, 1);
            }
            else {
                matrix.grid.RecalculateContentSize(newSource.GetLength(0), newSource.GetLength(1));
                matrix.grid.RecalculateCellSize(matrix.RenderSize);
            }
            matrix.RepaintLedsMask();
        }

        private static void OnSplitCellsChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            var matrix = (d as MatrixLED);
            if (matrix.Source == null || matrix.Source.Length < 1) {
                return;
            }
            matrix.mask = null;
            matrix.RepaintLedsMask();
        }

        private void RepaintLedsMask() {
            mask = BitmapHelper.Valid(mask, grid.ContentSize);
            if (SplitCells) {
                using (var context = new WriteableContext(mask)) {
                    context.WriteCells(Source, grid.CellSize, grid.Spacing);
                }
            }
            else {
                using (var context = new WriteableContext(mask)) {
                    context.WriteCells(Source, grid.CellSize + grid.Spacing, 0);
                }
            }            
        }

        protected override void OnRender(DrawingContext dc) {
            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));
            dc.PushOpacityMask(new ImageBrush(mask));
            dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));
        }

        protected override Size MeasureOverride(Size constraint)
            => MeasureArrangeHelper.ComputeSize(constraint, grid.ContentSize);

        protected override Size ArrangeOverride(Size arrangeSize)
            => MeasureArrangeHelper.ComputeSize(arrangeSize, grid.ContentSize);

        private sealed class Grid {

            private static readonly (int spacing, int cellSize) min;

            public int Spacing { get; private set; }
            public int CellSize { get; private set; }

            public int CountX { get; private set; }
            public int CountY { get; private set; }

            public Size ContentSize { get; private set; }

            public event Action Invalidate;

            static Grid() => min = (spacing: 1, cellSize: 4);
            public Grid() {
                Spacing = min.spacing;
                CellSize = min.cellSize;
                RecalculateContentSize(1, 1);
            }

            public void RecalculateContentSize(int newCountX, int newCountY) {
                if (CountX == newCountX && CountY == newCountY) {
                    return;
                }
                CountX = newCountX;
                CountY = newCountY;
                RecalculateContentSize();
                Invalidate?.Invoke();
            }

            public void RecalculateCellSize(Size available) {

                var minWidth = CalculateSideLength(CountX, min.cellSize, min.spacing);
                var scaleFactor = (int)Math.Round(available.Width / minWidth);

                var newCellSize = min.cellSize * scaleFactor;

                if (newCellSize < min.cellSize) {
                    newCellSize = min.cellSize;
                }
                if (newCellSize != CellSize) {
                    CellSize = newCellSize/* * 4*/;
                    RecalculateContentSize();
                }
            }

            private void RecalculateContentSize()
                => ContentSize = new Size(
                    CalculateSideLength(CountX, CellSize, Spacing),
                    CalculateSideLength(CountY, CellSize, Spacing));

            private int CalculateSideLength(int cellsCount, int cellSize, int spacing)
                => spacing + (cellSize + spacing) * cellsCount;
        }
    }
}