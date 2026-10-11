/*
 * Copyright (C) 2026 openEMS Project
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OPERATOR_SETUP_H
#define OPERATOR_SETUP_H

class Operator;
class Operator_Extension;
class Operator_Ext_TFSF;

// Construction only; the operator and its runtime interfaces keep their type.
class OperatorSetup
{
public:
	virtual ~OperatorSetup() {}
	virtual void CalcEC(Operator& op) = 0;
	virtual void CalcTimestep(Operator& op) = 0;
	virtual void CalcCoefficients(Operator& op) = 0;
	virtual void CalcPEC(Operator& op) = 0;
	virtual void SynchronizeOperator(Operator& op) = 0;
	virtual bool BuildTFSF(Operator_Ext_TFSF& extension, const double origin[3]) = 0;
	virtual void CompleteMur(unsigned int offset, unsigned int count, unsigned int startTimestep) = 0;
	virtual bool BuildExtension(Operator& op, Operator_Extension& extension) = 0;
	virtual double LumpedGeometry(int direction, const unsigned int start[3], const unsigned int stop[3]) = 0;
	virtual void ApplyLumpedRC(int direction, const unsigned int start[3], const unsigned int stop[3],
		double capacitance, double resistance, double epsilon, double kappa, bool caps) = 0;
	virtual void CompleteExcitation(bool magnetic, unsigned int count, unsigned int offset, unsigned int length, unsigned int period) = 0;
};

#endif
