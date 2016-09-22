﻿namespace CellUniverse.CustomControls.RapidCellularViewportDependencies {


    internal sealed class Settings {

        internal int SurfaceWidth        { get; private set; }
        internal int SurfaceHeight       { get; private set; }
        internal int SpacingBetweenCells { get; private set; }

        private int cellsHorizontal;
        private int cellsVertical;

        internal int CellSize { get; private set; }
        internal int OffsetX  { get; private set; }
        internal int OffsetY  { get; private set; }

        internal bool SizeChanged(int newSurfaceWidth, int newSurfaceHeight) {
            return
                newSurfaceWidth == SurfaceWidth && newSurfaceHeight == SurfaceHeight
                ? false : true;
        }

        internal bool CellsCountChanged(int newCellsHorizontal, int newCellsVertical) {
            return
                newCellsHorizontal == cellsHorizontal && newCellsVertical == cellsVertical
                ? false : true;
        }

        internal void Recalculate(
            int surfaceWidth, int surfaceHeight, int cellsHorizontal, int cellsVertical, int spacingBetweenCells = 1) {

            SurfaceWidth = surfaceWidth;
            SurfaceHeight = surfaceHeight;

            this.cellsHorizontal = cellsHorizontal;
            this.cellsVertical   = cellsVertical;
            SpacingBetweenCells  = spacingBetweenCells;

            CellSize = CalculateActualCellSize(
                cellsHorizontal, cellsVertical, surfaceWidth, surfaceHeight, spacingBetweenCells);

            OffsetX = CalculateActualOffsetToCenter(
                surfaceWidth, CalculateTotalInternalVectorLength(cellsHorizontal, CellSize, spacingBetweenCells));

            OffsetY = CalculateActualOffsetToCenter(
                surfaceHeight, CalculateTotalInternalVectorLength(cellsVertical, CellSize, spacingBetweenCells));

        }

        private int CalculateActualCellSize(
            int cellsHorizontal, int cellsVertical, int surfaceWidth, int surfaceHeight, int spacingBetweenCells) {
            int maxCellWidth  = CelculateMaxLengthOfSegmentsInAVector(cellsHorizontal, surfaceWidth,  spacingBetweenCells);
            int maxCellHeight = CelculateMaxLengthOfSegmentsInAVector(cellsVertical,   surfaceHeight, spacingBetweenCells);
            return maxCellHeight < maxCellWidth ? maxCellHeight : maxCellWidth;
        }

        private int CelculateMaxLengthOfSegmentsInAVector(int segmentsCount, int vectorLength, int spacingBetweenSegments) {
            int totalSpacing = (segmentsCount + 1) * spacingBetweenSegments;
            return (vectorLength - totalSpacing) / segmentsCount;
        }

        private int CalculateActualOffsetToCenter(int externalVectorLenght, int internalVectorLength) {
            return (externalVectorLenght - internalVectorLength) / 2;
        }

        private int CalculateTotalInternalVectorLength(int segmentsCount, int segmentLength, int spacingBetweenSegments) {
            int totalSegmentsLength = segmentsCount * segmentLength;
            int totalSpacing = (segmentsCount + 1) * spacingBetweenSegments;
            return totalSegmentsLength + totalSpacing;
        }
    }
}