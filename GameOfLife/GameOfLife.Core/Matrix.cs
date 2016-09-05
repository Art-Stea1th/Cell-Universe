using System;
using System.Collections.Generic;

namespace GameOfLife.Core {

    public class Matrix<T> : IEquatable<Matrix<T>> where T : IEquatable<T> {

        private T[][] _data;

        public int RowsCount { get { return _data.GetLength(0); } }
        public int ColumnsCount { get { return _data[0].GetLength(0); } }        

        public T this[int row, int column] {
            get {
                return _data[row][column];
            }
            set {
                _data[row][column] = value;
            }
        }

        // - Ctor

        public Matrix() {
            _data = TryAllocate(0, 0);
        }
        // TODO: Add copy
        public Matrix(T[][] array2d) {
            _data = TryAllocate(array2d.GetLength(0), array2d[0].GetLength(0));
        }

        public Matrix(int rows, int columns) {
            _data =
                IsValidSize(rows, columns)
                ? TryAllocate(rows, columns)
                : TryAllocate(0, 0);
        }

        // - Basics

        public static bool IsValidSize(int rows, int columns) {
            return rows > 0 && columns > 0 ? true : false;
        }

        private static T[][] TryAllocate(int rows, int columns) {
            try {
                T[][] result = new T[rows][];
                for (int row = 0; row < rows; row++) {
                    result[row] = new T[columns];
                }
                return result;
            }
            catch (OutOfMemoryException ex) { throw ex; }
        }
        // TODO: Add test
        public IEnumerable<IEnumerable<T>> Rows {
            get {
                for (int i = 0; i < RowsCount; i++) {
                    yield return GetRow(i);
                }
            }
        }
        // TODO: Add test
        public IEnumerable<IEnumerable<T>> Columns {
            get {
                for (int i = 0; i < ColumnsCount; i++) {
                    yield return GetColumn(i);
                }
            }
        }
        // TODO: Add test
        public IEnumerable<T> GetRow(int rowIndex) {
            for (int column = 0; column < ColumnsCount; column++) {
                yield return _data[rowIndex][column];
            }
        }
        // TODO: Add test
        public IEnumerable<T> GetColumn(int columnIndex) {
            for (int row = 0; row < RowsCount; row++) {
                yield return _data[row][columnIndex];
            }
        }
        // TODO: Add test
        public void CopyTo() {

        }
        // TODO: Add test
        public void AddRow() {
            throw new NotImplementedException();
        }
        // TODO: Add test
        public void AddColumn() {
            throw new NotImplementedException();
        }

        // - IEquatable<Matrix<T>>

        public bool Equals(Matrix<T> other) {

            if (RowsCount != other.RowsCount || ColumnsCount != other.ColumnsCount)
                return false;

            for (int row = 0; row < RowsCount; row++) {
                for (int column = 0; column < ColumnsCount; column++) {
                    if (!_data[row][column].Equals(other._data[row][column])) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}